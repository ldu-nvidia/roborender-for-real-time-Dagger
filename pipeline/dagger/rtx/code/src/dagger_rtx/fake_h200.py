from __future__ import annotations

import json
import math
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

import numpy as np

from .boundary import H200_COMMIT_PATH, H200_PREPARE_PATH, selection_receipt_body
from .contracts import (
    ACTION_DIM,
    CAPTURE_METADATA_KEYS,
    CONTRACT_VERSION,
    HOMIE_ACTION_DIM,
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_CAPTURE_VIEWS,
    RAVEN_VIEW_ORDER,
    ROUND_BETAS,
    STUDENT_ACTION_DIM,
    ArrayRule,
    ContractError,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
    canonical_json,
    canonical_wire_array,
    decode_array,
    decode_arrays,
    encode_array,
    payload_digest,
    require_identifier,
    sha256_bytes,
    validate_array,
    validate_beta,
)


_PREPARE_KEYS = {
    "contract_version", "idempotency_key", "step_key", "prepare_receipt",
    "capture_digest", "beta", "metadata", "arrays",
}
_METADATA_KEYS = set(CAPTURE_METADATA_KEYS) | {
    "contract_version", "step_key", "beta", "episode_assignment",
}
_ASSIGNMENT_KEYS = {
    "use_raytraced", "observation_source", "anchor_sha256", "assignment_source",
    "raven_source", "raven_commit",
}
_COMMIT_KEYS = {
    "contract_version", "idempotency_key", "step_key", "prepare_receipt", "selection_id",
    "selection_receipt", "commit_receipt", "transition", "sealed_window",
}
_TRANSITION_KEYS = {"reward", "done", "timeout", "metadata"}
_SEALED_KEYS = {
    "schema", "contract_version", "run_id", "round_id", "actor_id", "rollout_window_id",
    "row_count", "artifact_role", "learner_complete", "contains_policy_rgb",
    "authoritative_learner_chunk_owner", "episode_ids_are_per_step", "actual_dones_preserved",
    "no_previous_round_replay",
    "view_order", "per_step_identity_fields", "episode_ids", "episode_step_indices", "env_ids",
    "seqs", "round_ids", "observation_sources", "dones", "timeouts", "rewards",
    "behavior_sources", "prepare_receipts", "capture_digests", "selection_ids", "commit_digests",
    "commit_receipts", "path", "bytes", "sha256", "provenance",
}
_PROVENANCE_KEYS = {
    "raven_root", "raven_commit", "raven_videogen_trainer_sha256",
    "raven_teacher_homie_trainer_sha256", "isaac_capture_sha256", "image_dagger_config_sha256",
    "teacher_checkpoint_sha256", "homie_walk_sha256", "homie_stand_sha256", "episode_bindings",
}
_PER_STEP_FIELDS = [
    "episode_ids", "episode_step_indices", "rollout_step_indices", "round_ids", "env_id_per_step",
    "seq_per_step", "observation_source", "dones", "timeouts", "rewards", "behavior_source",
    "prepare_receipts", "capture_digests", "selection_ids", "commit_digests", "commit_receipts",
]


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def observation_input_digest(
    key: StepKey,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Digest the canonical, source-specific inputs used to form the policy observation.

    The fake cannot render RGB, so it commits to the exact inputs consumed by the
    H200 observation path.  Generated observations bind prompt, fixed anchors,
    h5_v4 depth, and robot controls.  Ray-traced observations bind the policy RGB
    triplet.  Array entries bind name, dtype, shape, and raw-byte SHA-256.
    """

    source = metadata["observation_source"]
    if source == "raytraced":
        names = [f"raytraced_rgb__{view}" for view in RAVEN_VIEW_ORDER]
    elif source == "generated_controls":
        names = []
        for view in RAVEN_VIEW_ORDER:
            names.extend(
                (
                    f"anchor_rgb__{view}",
                    f"depth_ctrl__{view}",
                    f"rgb_robot__{view}",
                )
            )
    else:  # validation normally catches this before digesting
        raise ContractError("unsupported observation_source")
    inputs = []
    for name in names:
        arr = canonical_wire_array(arrays[name])
        inputs.append(
            {
                "name": name,
                "dtype": arr.dtype.name,
                "shape": list(arr.shape),
                "sha256": sha256_bytes(arr.tobytes(order="C")),
            }
        )
    body = {
        "contract_version": CONTRACT_VERSION,
        "step_key": key.as_dict(),
        "observation_source": source,
        "prompt": metadata["prompt"],
        "view_order": list(RAVEN_VIEW_ORDER),
        "model_view_map": dict(MODEL_VIEW_MAP),
        "inputs": inputs,
    }
    return sha256_bytes(canonical_json(body))


class FakeH200State:
    def __init__(
        self,
        token: str,
        *,
        selection_draw: float = 0.99,
        shape_profile: RavenShapeProfile | None = None,
    ):
        if isinstance(selection_draw, bool) or not isinstance(selection_draw, (int, float)):
            raise ValueError("selection_draw must be numeric")
        if not math.isfinite(float(selection_draw)) or not 0 <= float(selection_draw) < 1:
            raise ValueError("selection_draw must be finite in [0,1)")
        self.token = token
        self.selection_draw = float(selection_draw)
        # Direct state-level unit tests may supply a deliberately tiny coherent
        # profile.  The HTTP server always resolves an omitted profile to the
        # production RavenShapeProfile before constructing this state.
        self.profile = shape_profile
        self.prepares: dict[str, tuple[str, dict[str, Any]]] = {}
        self.commits: dict[str, tuple[str, dict[str, Any]]] = {}
        self.commits_by_selection: dict[str, str] = {}
        self.prepare_calls = 0
        self.commit_calls = 0
        self.prepare_effects = 0
        self.commit_effects = 0
        self.fail_next_prepare = 0
        self.fail_next_commit = 0
        self.fail_after_next_prepare = 0
        self.fail_after_next_commit = 0
        self.last_prepare_request: dict[str, Any] | None = None
        self.last_commit_request: dict[str, Any] | None = None

    def durable_snapshot(self) -> dict[str, Any]:
        """Return the fake's durable H200 ledger in JSON-serializable form."""

        value = {
            "selection_draw": self.selection_draw,
            "prepares": self.prepares,
            "commits": self.commits,
            "commits_by_selection": self.commits_by_selection,
            "prepare_effects": self.prepare_effects,
            "commit_effects": self.commit_effects,
        }
        return json.loads(canonical_json(value))

    def restore_durable_snapshot(self, value: Mapping[str, Any]) -> None:
        expected = {
            "selection_draw", "prepares", "commits", "commits_by_selection",
            "prepare_effects", "commit_effects",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ContractError("invalid fake H200 durable snapshot")
        draw = value["selection_draw"]
        if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not 0 <= float(draw) < 1:
            raise ContractError("invalid persisted selection_draw")
        self.selection_draw = float(draw)
        for ledger_name in ("prepares", "commits"):
            ledger = value[ledger_name]
            if not isinstance(ledger, Mapping):
                raise ContractError(f"invalid persisted {ledger_name} ledger")
            restored: dict[str, tuple[str, dict[str, Any]]] = {}
            for idem, pair in ledger.items():
                require_identifier(idem, f"persisted {ledger_name} idempotency key")
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[1], Mapping):
                    raise ContractError(f"invalid persisted {ledger_name} entry")
                _require_sha256(pair[0], f"persisted {ledger_name} request digest")
                restored[idem] = (pair[0], dict(pair[1]))
            setattr(self, ledger_name, restored)
        mapping = value["commits_by_selection"]
        if not isinstance(mapping, Mapping):
            raise ContractError("invalid persisted selection commit map")
        self.commits_by_selection = {}
        for selection_id, idem in mapping.items():
            require_identifier(selection_id, "persisted selection_id")
            require_identifier(idem, "persisted commit idempotency key")
            if idem not in self.commits:
                raise ContractError("persisted selection map refers to missing commit")
            self.commits_by_selection[selection_id] = idem
        for name in ("prepare_effects", "commit_effects"):
            count = value[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ContractError(f"invalid persisted {name}")
            setattr(self, name, count)

    def _validate_prepare(self, request: Mapping[str, Any]) -> tuple[StepKey, str, dict[str, np.ndarray]]:
        if not isinstance(request, Mapping) or set(request) != _PREPARE_KEYS:
            raise ContractError("prepare request key mismatch")
        if request["contract_version"] != CONTRACT_VERSION:
            raise ContractError("prepare contract version mismatch")
        key = StepKey.from_mapping(request["step_key"])
        idem = require_identifier(request["idempotency_key"], "idempotency_key")
        beta = validate_beta(key.round_id, request["beta"])
        _require_sha256(request["prepare_receipt"], "prepare_receipt")
        _require_sha256(request["capture_digest"], "capture_digest")

        metadata = request["metadata"]
        if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
            raise ContractError("prepare metadata key mismatch")
        if metadata["contract_version"] != CONTRACT_VERSION:
            raise ContractError("metadata contract version mismatch")
        if metadata["step_key"] != key.as_dict():
            raise ContractError("metadata step_key does not match request step_key")
        if validate_beta(key.round_id, metadata["beta"]) != beta:
            raise ContractError("metadata beta does not match request beta")
        if metadata["view_order"] != list(RAVEN_VIEW_ORDER):
            raise ContractError("metadata view_order mismatch")
        if metadata["model_view_map"] != MODEL_VIEW_MAP:
            raise ContractError("metadata model_view_map mismatch")
        if metadata["depth_units"] != "meter":
            raise ContractError("metadata depth_units mismatch")
        if metadata["depth_control_encoding"] != "raven_h5_v4":
            raise ContractError("metadata depth control is not pinned Raven h5_v4")
        if metadata["robot_control_encoding"] != "textured_robot_rgb_on_black":
            raise ContractError("metadata robot control encoding mismatch")
        if not isinstance(metadata["prompt"], str) or not metadata["prompt"] or len(metadata["prompt"]) > 4096:
            raise ContractError("metadata prompt must be non-empty and at most 4096 characters")
        for name in ("env_id", "seq"):
            if isinstance(metadata[name], bool) or not isinstance(metadata[name], int):
                raise ContractError(f"metadata {name} must be an integer")

        assignment = metadata["episode_assignment"]
        if not isinstance(assignment, Mapping) or set(assignment) != _ASSIGNMENT_KEYS:
            raise ContractError("episode_assignment key mismatch")
        source = metadata["observation_source"]
        if source not in ("raytraced", "generated_controls"):
            raise ContractError("unsupported observation_source")
        expected_raytraced = source == "raytraced"
        if assignment["use_raytraced"] is not expected_raytraced:
            raise ContractError("episode_assignment use_raytraced/source mismatch")
        if assignment["observation_source"] != source:
            raise ContractError("episode_assignment observation_source mismatch")
        if assignment["assignment_source"] != "pinned_raven_id_sample_anchor":
            raise ContractError("episode assignment source mismatch")
        if assignment["raven_source"] != PINNED_RAVEN_ROOT or assignment["raven_commit"] != PINNED_RAVEN_COMMIT:
            raise ContractError("episode assignment does not bind pinned Raven")
        _require_sha256(assignment["anchor_sha256"], "anchor_sha256")

        decoded = decode_arrays(request["arrays"])
        profile = self.profile
        if profile is None:
            # Compatibility for direct state-level tests only.  Cross-array
            # relationships are still validated; the externally reachable fake
            # always has the explicit production profile.
            ray_sample = next(
                (decoded[name] for name in decoded if name.startswith("raytraced_rgb__")),
                None,
            )
            profile = RavenShapeProfile(
                actor_obs=decoded["actor_obs"].shape[0],
                teacher_obs=decoded["teacher_obs"].shape[0],
                critic_obs=decoded["critic_obs"].shape[0],
                homie_obs=decoded["homie_obs"].shape[0],
                depth_h=decoded["depth_metric__head_color"].shape[0],
                depth_w=decoded["depth_metric__head_color"].shape[1],
                control_h=decoded["depth_ctrl__head_color"].shape[0],
                control_w=decoded["depth_ctrl__head_color"].shape[1],
                policy_h=ray_sample.shape[0] if ray_sample is not None else 108,
                policy_w=ray_sample.shape[1] if ray_sample is not None else 192,
            )
        base_rules = profile.capture_rules()
        ray_names = {f"raytraced_rgb__{view}" for view in RAVEN_CAPTURE_VIEWS}
        expected_names = set(base_rules) | (ray_names if expected_raytraced else set())
        if set(decoded) != expected_names:
            missing = sorted(expected_names - set(decoded))
            unexpected = sorted(set(decoded) - expected_names)
            raise ContractError(f"prepare arrays mismatch; missing={missing}, unexpected={unexpected}")
        for name, rule in base_rules.items():
            validate_array(name, decoded[name], rule)
        if expected_raytraced:
            ray_rule = ArrayRule((profile.policy_h, profile.policy_w, 3), "float32", True)
            for name in ray_names:
                ray = validate_array(name, decoded[name], ray_rule)
                if bool(np.any(ray < 0)) or bool(np.any(ray > 1)):
                    raise ContractError(f"{name} values must be in [0,1]")
        teacher = decoded["teacher_action"]
        homie = decoded["homie_action"]
        if not np.array_equal(teacher[STUDENT_ACTION_DIM:], homie):
            raise ContractError("teacher_action[11:] does not exactly equal homie_action")
        if anchor_rgb_digest(decoded) != assignment["anchor_sha256"]:
            raise ContractError("anchor RGB digest mismatch")
        if payload_digest(metadata, decoded) != request["capture_digest"]:
            raise ContractError("capture_digest does not match canonical metadata and arrays")
        return key, idem, decoded

    def prepare(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.prepare_calls += 1
        self.last_prepare_request = dict(request)
        key, idem, decoded = self._validate_prepare(request)
        body_digest = sha256_bytes(canonical_json(request))
        prior = self.prepares.get(idem)
        if prior:
            if prior[0] != body_digest:
                raise RuntimeError("idempotency payload conflict")
            return prior[1]
        teacher = decoded["teacher_action"]
        homie = decoded["homie_action"]
        student11 = np.linspace(-1, 1, STUDENT_ACTION_DIM, dtype=np.float32)
        beta = validate_beta(key.round_id, request["beta"])
        source = "teacher" if self.selection_draw < beta else "student"
        selected = teacher if source == "teacher" else np.concatenate((student11, homie)).astype(np.float32)
        response = {
            "contract_version": CONTRACT_VERSION,
            "idempotency_key": idem,
            "step_key": key.as_dict(),
            "prepare_receipt": request["prepare_receipt"],
            "capture_digest": request["capture_digest"],
            "beta": beta,
            "selection_id": f"selection-{body_digest[:48]}",
            "selection_draw": self.selection_draw,
            "selected_source": source,
            "student_action_11": encode_array(student11),
            "selected_action": encode_array(selected),
            "observation_digest": observation_input_digest(key, request["metadata"], decoded),
        }
        response["selection_receipt"] = sha256_bytes(canonical_json(selection_receipt_body(response)))
        self.prepares[idem] = (body_digest, response)
        self.prepare_effects += 1
        return response

    def _selection_for_commit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        matches = [
            response
            for _, response in self.prepares.values()
            if response["selection_id"] == request["selection_id"]
        ]
        if len(matches) != 1:
            raise ContractError("commit selection_id has no unique prepared selection")
        selection = matches[0]
        for name in ("step_key", "prepare_receipt", "selection_receipt"):
            if request[name] != selection[name]:
                raise ContractError(f"commit {name} does not bind the prepared selection")
        expected_selection_receipt = sha256_bytes(canonical_json(selection_receipt_body(selection)))
        if request["selection_receipt"] != expected_selection_receipt:
            raise ContractError("commit selection_receipt digest mismatch")
        return selection

    @staticmethod
    def _validate_transition(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _TRANSITION_KEYS:
            raise ContractError("transition key mismatch")
        reward = value["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ContractError("transition reward must be finite")
        if not isinstance(value["done"], bool) or not isinstance(value["timeout"], bool):
            raise ContractError("transition done and timeout must be booleans")
        if not isinstance(value["metadata"], Mapping):
            raise ContractError("transition metadata must be an object")
        canonical_json(value["metadata"])
        return dict(value)

    @staticmethod
    def _validate_sealed_window(value: Any, key: StepKey) -> None:
        if not isinstance(value, Mapping) or set(value) != _SEALED_KEYS:
            raise ContractError("sealed_window key mismatch")
        constants = {
            "schema": "roborender.raven.dagger-window.v1",
            "contract_version": CONTRACT_VERSION,
            "run_id": key.run_id,
            "round_id": key.round_id,
            "actor_id": key.actor_id,
            "rollout_window_id": key.rollout_window_id,
            "row_count": 81,
            "artifact_role": "rtx_identity_outcome_audit",
            "learner_complete": False,
            "contains_policy_rgb": False,
            "authoritative_learner_chunk_owner": "h200_render_student_ledger",
            "episode_ids_are_per_step": True,
            "actual_dones_preserved": True,
            "no_previous_round_replay": True,
            "view_order": list(RAVEN_VIEW_ORDER),
            "per_step_identity_fields": _PER_STEP_FIELDS,
            "path": "sealed.npz",
        }
        for name, expected in constants.items():
            if value[name] != expected:
                raise ContractError(f"sealed_window {name} mismatch")
        if isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] < 1:
            raise ContractError("sealed_window bytes must be a positive integer")
        _require_sha256(value["sha256"], "sealed_window sha256")
        list_rules = {
            "episode_ids": lambda item: isinstance(item, str) and bool(item),
            "episode_step_indices": lambda item: isinstance(item, int) and not isinstance(item, bool) and item >= 0,
            "env_ids": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "seqs": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "round_ids": lambda item: isinstance(item, int) and not isinstance(item, bool) and item == key.round_id,
            "observation_sources": lambda item: item in ("raytraced", "generated_controls"),
            "dones": lambda item: isinstance(item, bool),
            "timeouts": lambda item: isinstance(item, bool),
            "rewards": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)),
            "behavior_sources": lambda item: item in ("teacher", "student"),
            "prepare_receipts": lambda item: isinstance(item, str) and len(item) == 64 and not any(c not in "0123456789abcdef" for c in item),
            "capture_digests": lambda item: isinstance(item, str) and len(item) == 64 and not any(c not in "0123456789abcdef" for c in item),
            "selection_ids": lambda item: isinstance(item, str) and bool(item) and len(item) <= 128,
            "commit_digests": lambda item: isinstance(item, str) and len(item) == 64 and not any(c not in "0123456789abcdef" for c in item),
            "commit_receipts": lambda item: isinstance(item, str) and len(item) == 64 and not any(c not in "0123456789abcdef" for c in item),
        }
        for name, predicate in list_rules.items():
            items = value[name]
            if not isinstance(items, list) or len(items) != 81 or not all(predicate(item) for item in items):
                raise ContractError(f"sealed_window {name} must contain 81 valid entries")
        for item in value["episode_ids"]:
            require_identifier(item, "sealed_window episode_id")
        for item in value["selection_ids"]:
            require_identifier(item, "sealed_window selection_id")
        for name in ("prepare_receipts", "capture_digests", "commit_digests", "commit_receipts"):
            for item in value[name]:
                _require_sha256(item, f"sealed_window {name}")
        for index in range(1, 81):
            same_episode = value["episode_ids"][index] == value["episode_ids"][index - 1]
            if same_episode:
                if value["episode_step_indices"][index] != value["episode_step_indices"][index - 1] + 1:
                    raise ContractError("sealed_window episode steps are not contiguous")
                if value["dones"][index - 1] or value["timeouts"][index - 1]:
                    raise ContractError("sealed_window continued an episode after done/timeout")
                if value["observation_sources"][index] != value["observation_sources"][index - 1]:
                    raise ContractError("sealed_window observation source changed within an episode")
            else:
                if not (value["dones"][index - 1] or value["timeouts"][index - 1]):
                    raise ContractError("sealed_window changed episode without done/timeout")
                if value["episode_step_indices"][index] != 0:
                    raise ContractError("sealed_window new episode did not reset episode_step_index")
        if value["episode_ids"][80] != key.episode_id or value["episode_step_indices"][80] != key.episode_step_index:
            raise ContractError("sealed_window final per-step identity does not match commit step")
        provenance = value["provenance"]
        if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_KEYS:
            raise ContractError("sealed_window provenance key mismatch")
        if provenance["raven_root"] != PINNED_RAVEN_ROOT or provenance["raven_commit"] != PINNED_RAVEN_COMMIT:
            raise ContractError("sealed_window provenance does not bind pinned Raven")
        for name in _PROVENANCE_KEYS - {"raven_root", "raven_commit", "episode_bindings"}:
            _require_sha256(provenance[name], f"provenance {name}")
        bindings = provenance["episode_bindings"]
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 81:
            raise ContractError("sealed_window episode_bindings must contain 1..81 entries")
        expected_binding_indices = [0] + [
            index
            for index in range(1, 81)
            if value["episode_ids"][index] != value["episode_ids"][index - 1]
        ]
        if len(bindings) != len(expected_binding_indices):
            raise ContractError("sealed_window episode_bindings do not cover every episode transition")
        binding_keys = {
            "episode_id", "first_rollout_step_index", "use_raytraced", "observation_source", "anchor_sha256",
        }
        for binding, index in zip(bindings, expected_binding_indices):
            if not isinstance(binding, Mapping) or set(binding) != binding_keys:
                raise ContractError("sealed_window episode binding key mismatch")
            source = value["observation_sources"][index]
            if (
                binding["episode_id"] != value["episode_ids"][index]
                or binding["first_rollout_step_index"] != index
                or binding["observation_source"] != source
                or binding["use_raytraced"] is not (source == "raytraced")
            ):
                raise ContractError("sealed_window episode binding does not match per-step identity")
            require_identifier(binding["episode_id"], "episode binding episode_id")
            _require_sha256(binding["anchor_sha256"], "episode binding anchor_sha256")

    @staticmethod
    def _expected_commit_receipt(selection: Mapping[str, Any], transition: Mapping[str, Any]) -> str:
        commit_body = {
            "step_key": selection["step_key"],
            "prepare_receipt": selection["prepare_receipt"],
            "selection_id": selection["selection_id"],
            "selected_source": selection["selected_source"],
            "beta": selection["beta"],
            "selected_action": selection["selected_action"],
        }
        commit_digest = sha256_bytes(canonical_json(commit_body))
        return sha256_bytes(canonical_json({"commit_digest": commit_digest, "result": transition}))

    def commit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.commit_calls += 1
        self.last_commit_request = dict(request)
        if not isinstance(request, Mapping) or set(request) != _COMMIT_KEYS:
            raise ContractError("commit request key mismatch")
        if request["contract_version"] != CONTRACT_VERSION:
            raise ContractError("commit contract version mismatch")
        key = StepKey.from_mapping(request["step_key"])
        idem = require_identifier(request["idempotency_key"], "idempotency_key")
        _require_sha256(request["prepare_receipt"], "prepare_receipt")
        _require_sha256(request["selection_receipt"], "selection_receipt")
        _require_sha256(request["commit_receipt"], "commit_receipt")
        require_identifier(request["selection_id"], "selection_id")
        body_digest = sha256_bytes(canonical_json(request))
        prior = self.commits.get(idem)
        if prior:
            if prior[0] != body_digest:
                raise RuntimeError("idempotency payload conflict")
            return prior[1]
        prior_idem = self.commits_by_selection.get(request["selection_id"])
        if prior_idem is not None:
            raise RuntimeError(f"selection already committed under idempotency key {prior_idem}")
        selection = self._selection_for_commit(request)
        transition = self._validate_transition(request["transition"])
        if key.rollout_step_index == 80:
            self._validate_sealed_window(request["sealed_window"], key)
            sealed = request["sealed_window"]
            expected_commit_digest = sha256_bytes(
                canonical_json(
                    {
                        "step_key": selection["step_key"],
                        "prepare_receipt": selection["prepare_receipt"],
                        "selection_id": selection["selection_id"],
                        "selected_source": selection["selected_source"],
                        "beta": selection["beta"],
                        "selected_action": selection["selected_action"],
                    }
                )
            )
            final_bindings = {
                "dones": transition["done"],
                "timeouts": transition["timeout"],
                "rewards": float(transition["reward"]),
                "behavior_sources": selection["selected_source"],
                "prepare_receipts": selection["prepare_receipt"],
                "capture_digests": selection["capture_digest"],
                "selection_ids": selection["selection_id"],
                "commit_digests": expected_commit_digest,
                "commit_receipts": request["commit_receipt"],
            }
            for name, expected in final_bindings.items():
                if sealed[name][80] != expected:
                    raise ContractError(f"sealed_window final {name} does not match committed selection/outcome")
        elif request["sealed_window"] is not None:
            raise ContractError("sealed_window is only permitted at rollout_step_index 80")
        if request["commit_receipt"] != self._expected_commit_receipt(selection, transition):
            raise ContractError("commit_receipt does not bind selection and transition")
        response = {
            "contract_version": CONTRACT_VERSION,
            "idempotency_key": idem,
            "step_key": request["step_key"],
            "commit_receipt": request["commit_receipt"],
            "status": "committed",
        }
        response["remote_commit_receipt"] = sha256_bytes(canonical_json(response))
        self.commits[idem] = (body_digest, response)
        self.commits_by_selection[request["selection_id"]] = idem
        self.commit_effects += 1
        return response


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeH200Contract/1"

    def log_message(self, format, *args):
        return

    def do_POST(self):
        state: FakeH200State = self.server.state  # type: ignore[attr-defined]
        if self.headers.get("Authorization") != f"Bearer {state.token}":
            return self._reply(401, {"error": "unauthorized"})
        if self.headers.get_content_type() != "application/json":
            return self._reply(415, {"error": "application_json_required"})
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            return self._reply(411, {"error": "content_length_required"})
        length = int(length_text)
        if length > 64 << 20:
            return self._reply(413, {"error": "body_too_large"})
        body = self.rfile.read(length)
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ContractError("request JSON must be an object")
            if body != canonical_json(request):
                raise ContractError("request JSON must use the frozen canonical encoding")
            if self.path == H200_PREPARE_PATH:
                if state.fail_next_prepare:
                    state.fail_next_prepare -= 1
                    return self._reply(503, {"error": "injected_prepare_failure"})
                response = state.prepare(request)
                if state.fail_after_next_prepare:
                    state.fail_after_next_prepare -= 1
                    return self._reply(503, {"error": "injected_post_effect_prepare_failure"})
            elif self.path == H200_COMMIT_PATH:
                if state.fail_next_commit:
                    state.fail_next_commit -= 1
                    return self._reply(503, {"error": "injected_commit_failure"})
                response = state.commit(request)
                if state.fail_after_next_commit:
                    state.fail_after_next_commit -= 1
                    return self._reply(503, {"error": "injected_post_effect_commit_failure"})
            else:
                return self._reply(404, {"error": "not_found"})
        except RuntimeError as exc:
            return self._reply(409, {"error": "conflict", "message": str(exc)})
        except Exception as exc:
            return self._reply(400, {"error": "invalid", "message": str(exc)})
        self._reply(200, response)

    def _reply(self, status: int, value: Mapping[str, Any]):
        body = canonical_json(value)
        if self.server.response_trailing_newline:  # type: ignore[attr-defined]
            body += b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeH200Server(AbstractContextManager):
    def __init__(
        self,
        token: str = "test-token",
        *,
        selection_draw: float = 0.99,
        response_trailing_newline: bool = False,
        shape_profile: RavenShapeProfile | None = None,
        durable_snapshot: Mapping[str, Any] | None = None,
    ):
        self.state = FakeH200State(
            token,
            selection_draw=selection_draw,
            shape_profile=shape_profile or RavenShapeProfile(),
        )
        if durable_snapshot is not None:
            self.state.restore_durable_snapshot(durable_snapshot)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self.state  # type: ignore[attr-defined]
        self.server.response_trailing_newline = response_trailing_newline  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False
