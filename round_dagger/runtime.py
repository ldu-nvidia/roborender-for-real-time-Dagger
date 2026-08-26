"""Additive runtime contracts for the round-based RoboRender DAgger actor.

The production actor predates these interfaces and remains the source of
truth.  This module provides a small, synchronous facade that can observe the
legacy rollout in ``shadow`` mode without owning environment stepping, policy
evaluation, beta sampling, reset handling, or shard publication.

The contracts intentionally have no Isaac Lab or PyTorch import.  Ordinary
subscribers receive value-free metadata, never tensor or NumPy payloads.
"""

from __future__ import annotations

import os
import random
import sys
import threading
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence


RUNTIME_ADAPTER_MODE_ENV = "ROUND_DAGGER_RUNTIME_ADAPTER_MODE"
RUNTIME_ADAPTER_SHADOW_ACK_ENV = "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK"
RUNTIME_ADAPTER_SHADOW_ACK = "metadata-only-v1"
RUNTIME_ADAPTER_VALIDATION_ACK_ENV = (
    "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK"
)
RUNTIME_ADAPTER_VALIDATION_ACK = "paired-1x2-v1"
RUNTIME_ADAPTER_EVIDENCE_ENV = "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE"
RUNTIME_ADAPTER_INPUT_PROFILE_ENV = (
    "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE"
)
RUNTIME_ADAPTER_INPUT_PROFILE_LIVE = "live-isaac-v1"
RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_ZERO_MASK = (
    "canonical-zero-robot-mask-v1"
)
RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA = (
    "canonical-zero-robot-mask-deterministic-cuda-v1"
)
RUNTIME_ADAPTER_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_RUNTIME_ADAPTER_MODES = frozenset({"legacy", "shadow"})
_RUNTIME_ADAPTER_INPUT_PROFILES = frozenset(
    {
        RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
        RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_ZERO_MASK,
        RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA,
    }
)


class RuntimeAdapterError(RuntimeError):
    """A runtime adapter contract was violated."""


class SubscriberExecutionError(RuntimeAdapterError):
    """A synchronous runtime subscriber failed."""


class SubscriberSideEffectError(RuntimeAdapterError):
    """A subscriber changed actor-global state such as an RNG."""


@dataclass(frozen=True)
class ArrayMetadata:
    """Immutable, value-free description exposed to ordinary subscribers."""

    type_name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class OpaqueMetadata:
    """Value-free description for a non-array object."""

    type_name: str


def runtime_adapter_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the explicitly selected adapter mode.

    ``legacy`` is deliberately the default.  Phase one only supports a
    read-only ``shadow`` observer; there is no mode in which these adapters own
    the validated production rollout loop.
    """

    source = os.environ if environment is None else environment
    raw = source.get(RUNTIME_ADAPTER_MODE_ENV, "legacy")
    mode = str(raw).strip().lower()
    if mode not in _RUNTIME_ADAPTER_MODES:
        allowed = ", ".join(sorted(_RUNTIME_ADAPTER_MODES))
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_MODE_ENV} must be one of {allowed}; got {raw!r}"
        )
    if mode == "shadow":
        acknowledgement = source.get(RUNTIME_ADAPTER_SHADOW_ACK_ENV)
        if acknowledgement != RUNTIME_ADAPTER_SHADOW_ACK:
            raise RuntimeAdapterError(
                f"shadow mode requires {RUNTIME_ADAPTER_SHADOW_ACK_ENV}="
                f"{RUNTIME_ADAPTER_SHADOW_ACK}"
            )
    return mode


def runtime_adapter_validation_evidence_path(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Validate the second, bounded A/B authorization and evidence path.

    This gate is intentionally separate from the generic metadata-only shadow
    acknowledgement.  It prevents an ambient shadow flag from publishing a
    validation marker and confines the marker to the actor's explicit run
    directory.  Filesystem type and no-overwrite checks remain the trainer's
    responsibility immediately before its atomic publication.
    """

    source = os.environ if environment is None else environment
    acknowledgement = source.get(RUNTIME_ADAPTER_VALIDATION_ACK_ENV)
    if acknowledgement != RUNTIME_ADAPTER_VALIDATION_ACK:
        raise RuntimeAdapterError(
            "runtime validation requires "
            f"{RUNTIME_ADAPTER_VALIDATION_ACK_ENV}="
            f"{RUNTIME_ADAPTER_VALIDATION_ACK}"
        )
    raw_run_dir = source.get("TRAIN_RUN_DIR")
    raw_evidence = source.get(RUNTIME_ADAPTER_EVIDENCE_ENV)
    for name, value in (
        ("TRAIN_RUN_DIR", raw_run_dir),
        (RUNTIME_ADAPTER_EVIDENCE_ENV, raw_evidence),
    ):
        if not value or "\x00" in str(value) or not os.path.isabs(str(value)):
            raise RuntimeAdapterError(f"{name} must be a non-empty absolute path")
        if os.path.normpath(str(value)) != str(value):
            raise RuntimeAdapterError(f"{name} must be lexically canonical")
    run_dir = str(raw_run_dir)
    evidence = str(raw_evidence)
    if not evidence.endswith(".json") or os.path.basename(evidence) == ".json":
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_EVIDENCE_ENV} must name a JSON file"
        )
    try:
        common = os.path.commonpath((run_dir, evidence))
    except ValueError as exc:
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_EVIDENCE_ENV} is outside TRAIN_RUN_DIR"
        ) from exc
    if common != run_dir or evidence == run_dir:
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_EVIDENCE_ENV} must be inside TRAIN_RUN_DIR"
        )
    return evidence


def runtime_adapter_validation_input_profile(
    environment: Mapping[str, str] | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Return the explicitly authorized bounded-validation input profile.

    The canonical-mask profile exists only to isolate adapter semantics from
    cross-process Isaac semantic-mask rasterization variance.  It is never a
    production fallback: setting the profile without the exact bounded A/B
    acknowledgement fails closed.
    """

    source = os.environ if environment is None else environment
    raw = source.get(RUNTIME_ADAPTER_INPUT_PROFILE_ENV)
    if raw is None or not str(raw).strip():
        if required:
            raise RuntimeAdapterError(
                "runtime validation requires "
                f"{RUNTIME_ADAPTER_INPUT_PROFILE_ENV}"
            )
        return None
    if source.get(RUNTIME_ADAPTER_VALIDATION_ACK_ENV) != RUNTIME_ADAPTER_VALIDATION_ACK:
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_INPUT_PROFILE_ENV} is validation-only and requires "
            f"{RUNTIME_ADAPTER_VALIDATION_ACK_ENV}="
            f"{RUNTIME_ADAPTER_VALIDATION_ACK}"
        )
    profile = str(raw).strip().lower()
    if profile not in _RUNTIME_ADAPTER_INPUT_PROFILES:
        allowed = ", ".join(sorted(_RUNTIME_ADAPTER_INPUT_PROFILES))
        raise RuntimeAdapterError(
            f"{RUNTIME_ADAPTER_INPUT_PROFILE_ENV} must be one of {allowed}; "
            f"got {raw!r}"
        )
    return profile


@dataclass(frozen=True)
class EnvironmentStep:
    """The unmodified result of one vector-environment step."""

    observations: MutableMapping[str, Any]
    rewards: Any
    dones: Any
    infos: Mapping[str, Any]


class VectorEnvironment(Protocol):
    """Minimal vector-environment surface required by the actor."""

    @property
    def num_envs(self) -> int: ...

    def reset_all(self) -> MutableMapping[str, Any]: ...

    def step(self, actor_state: Mapping[str, Any]) -> Any: ...


class VectorEnvironmentAdapter:
    """Typed pass-through facade over the existing Isaac environment.

    Isaac already performs partial resets inside ``step``.  This adapter must
    never call a reset method in response to ``dones``.
    """

    def __init__(self, environment: VectorEnvironment):
        self._environment = environment

    @property
    def raw_environment(self) -> VectorEnvironment:
        return self._environment

    @property
    def num_envs(self) -> int:
        value = int(self._environment.num_envs)
        if value <= 0:
            raise RuntimeAdapterError("vector environment must contain at least one env")
        return value

    def reset_all(self) -> MutableMapping[str, Any]:
        observations = self._environment.reset_all()
        if not isinstance(observations, MutableMapping):
            raise RuntimeAdapterError("reset_all() must return a mutable observation mapping")
        return observations

    def step(self, actor_state: Mapping[str, Any]) -> EnvironmentStep:
        result = self._environment.step(actor_state)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeAdapterError(
                "environment.step() must return (observations, rewards, dones, infos)"
            )
        observations, rewards, dones, infos = result
        if not isinstance(observations, MutableMapping):
            raise RuntimeAdapterError("step observations must be a mutable mapping")
        if not isinstance(infos, Mapping):
            raise RuntimeAdapterError("step infos must be a mapping")
        return EnvironmentStep(observations, rewards, dones, infos)


class ObservationTransform(Protocol):
    """A pre-policy observation transform."""

    def apply(
        self,
        observations: MutableMapping[str, Any],
        dones: Any | None = None,
    ) -> MutableMapping[str, Any]: ...


class RoboRenderObservationTransform:
    """Adapt the existing RoboRender relabel function without changing identity.

    The legacy PPO loop stores the original observation dictionary after the
    policy call.  Installing generated RGB in-place is therefore required to
    keep the policy input and immutable replay row identical.
    """

    def __init__(
        self,
        relabel: Callable[[MutableMapping[str, Any], Any | None], Mapping[str, Any]],
        *,
        vision_key: str = "vision_obs",
    ):
        self._relabel = relabel
        self._vision_key = str(vision_key)
        if not self._vision_key:
            raise RuntimeAdapterError("vision_key must be non-empty")

    def apply(
        self,
        observations: MutableMapping[str, Any],
        dones: Any | None = None,
    ) -> MutableMapping[str, Any]:
        if self._vision_key not in observations:
            raise RuntimeAdapterError(
                f"RoboRender input is missing {self._vision_key!r}"
            )
        relabelled = self._relabel(observations, dones)
        if not isinstance(relabelled, Mapping) or self._vision_key not in relabelled:
            raise RuntimeAdapterError(
                f"RoboRender output is missing {self._vision_key!r}"
            )
        previous = observations[self._vision_key]
        generated = relabelled[self._vision_key]
        previous_shape = getattr(previous, "shape", None)
        generated_shape = getattr(generated, "shape", None)
        if previous_shape is None or generated_shape is None:
            raise RuntimeAdapterError("RoboRender vision values must expose shape")
        if tuple(previous_shape) != tuple(generated_shape):
            raise RuntimeAdapterError(
                "RoboRender changed policy observation shape: "
                f"{tuple(previous_shape)} -> {tuple(generated_shape)}"
            )
        observations[self._vision_key] = generated
        return observations


@dataclass(frozen=True)
class ActionDecision:
    """One already-computed DAgger decision.

    The adapter consumes the final legacy policy-state mapping.  It never calls
    a policy or samples beta, so the behavior action and expert mask remain the
    exact policy-space values passed to ``env.step`` and committed to replay.
    Environment-internal delta integration, clipping, and control delay are a
    later actuator boundary and are deliberately not relabeled as the DAgger
    behavior decision.
    """

    student_actions: Any
    expert_actions: Any
    behavior_actions: Any
    expert_mask: Any
    beta: float
    policy_state: Mapping[str, Any]

    @classmethod
    def from_policy_state(
        cls,
        policy_state: Mapping[str, Any],
        *,
        beta: float,
    ) -> "ActionDecision":
        required = (
            "student_actions",
            "gt_actions",
            "actions",
            "expert_mask",
        )
        missing = tuple(key for key in required if key not in policy_state)
        if missing:
            raise RuntimeAdapterError(
                "DAgger policy state is missing: " + ", ".join(missing)
            )
        parsed_beta = float(beta)
        if not 0.0 <= parsed_beta <= 1.0:
            raise RuntimeAdapterError(f"DAgger beta must be in [0, 1], got {beta!r}")
        frozen_state = MappingProxyType(dict(policy_state))
        decision = cls(
            student_actions=frozen_state["student_actions"],
            expert_actions=frozen_state["gt_actions"],
            behavior_actions=frozen_state["actions"],
            expert_mask=frozen_state["expert_mask"],
            beta=parsed_beta,
            policy_state=frozen_state,
        )
        decision.validate_shapes()
        return decision

    def validate_shapes(self) -> None:
        """Validate batch and action shapes without reading tensor contents."""

        action_shapes = tuple(
            getattr(value, "shape", None)
            for value in (
                self.student_actions,
                self.expert_actions,
                self.behavior_actions,
            )
        )
        if any(shape is None for shape in action_shapes):
            raise RuntimeAdapterError("DAgger actions must expose shape")
        normalized = tuple(tuple(int(dim) for dim in shape) for shape in action_shapes)
        if len(set(normalized)) != 1:
            raise RuntimeAdapterError(
                "student, expert, and behavior action shapes differ: "
                f"{normalized}"
            )
        if not normalized[0]:
            raise RuntimeAdapterError("DAgger actions must contain a batch dimension")
        mask_shape = getattr(self.expert_mask, "shape", None)
        if mask_shape is None or not tuple(mask_shape):
            raise RuntimeAdapterError("expert_mask must expose a batch dimension")
        if int(mask_shape[0]) != normalized[0][0]:
            raise RuntimeAdapterError(
                "expert_mask and action batch dimensions differ: "
                f"{tuple(mask_shape)} vs {normalized[0]}"
            )


class DaggerAgent(Protocol):
    """Agent surface that returns complete DAgger action lineage."""

    def decide(
        self,
        observations: MutableMapping[str, Any],
        dones: Any | None = None,
    ) -> ActionDecision: ...


class DaggerAgentAdapter:
    """Adapt one bound legacy policy call into an :class:`ActionDecision`."""

    def __init__(
        self,
        policy_step: Callable[
            [MutableMapping[str, Any], Any | None], Mapping[str, Any]
        ],
        beta: Callable[[], float] | float,
    ):
        self._policy_step = policy_step
        self._beta = beta

    def decide(
        self,
        observations: MutableMapping[str, Any],
        dones: Any | None = None,
    ) -> ActionDecision:
        policy_state = self._policy_step(observations, dones)
        if not isinstance(policy_state, Mapping):
            raise RuntimeAdapterError("policy_step must return a mapping")
        beta = self._beta() if callable(self._beta) else self._beta
        return ActionDecision.from_policy_state(policy_state, beta=float(beta))


@dataclass(frozen=True)
class RoundContext:
    """Immutable metadata for one actor rollout."""

    num_envs: int
    expected_steps: int
    metadata: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        num_envs: int,
        expected_steps: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RoundContext":
        if int(num_envs) <= 0 or int(expected_steps) <= 0:
            raise RuntimeAdapterError("round dimensions must be positive")
        return cls(
            num_envs=int(num_envs),
            expected_steps=int(expected_steps),
            metadata=MappingProxyType(dict(metadata or {})),
        )


@dataclass(frozen=True)
class DecisionRecord:
    """Pre-environment record for an already-finalized action decision."""

    rollout_step: int
    observations: Mapping[str, Any]
    decision: ActionDecision


@dataclass(frozen=True)
class StepRecord:
    """Canonical committed transition read from the legacy rollout storage."""

    rollout_step: int
    observations: Mapping[str, Any]
    decision: ActionDecision
    rewards: Any
    dones: Any
    time_outs: Any
    global_env_ids: Any
    episode_ids: Any
    episode_steps: Any
    observation_is_raytraced: Any
    infos: Mapping[str, Any]

    @classmethod
    def from_storage(
        cls,
        storage: Any,
        *,
        rollout_step: int,
        observation_keys: Sequence[str],
        beta: float,
        infos: Mapping[str, Any],
    ) -> "StepRecord":
        row = int(rollout_step)
        if row < 0:
            raise RuntimeAdapterError("rollout_step must be non-negative")

        def read(key: str) -> Any:
            try:
                values = storage.query_key(key)
                return values[row]
            except Exception as exc:
                raise RuntimeAdapterError(
                    f"canonical rollout storage is missing row {row} for {key!r}"
                ) from exc

        observations = MappingProxyType(
            {str(key): read(str(key)) for key in observation_keys}
        )
        policy_state = {
            "student_actions": read("student_actions"),
            "gt_actions": read("gt_actions"),
            "actions": read("actions"),
            "expert_mask": read("expert_mask"),
        }
        decision = ActionDecision.from_policy_state(policy_state, beta=beta)
        if not isinstance(infos, Mapping):
            raise RuntimeAdapterError("transition infos must be a mapping")
        return cls(
            rollout_step=row,
            observations=observations,
            decision=decision,
            rewards=read("rewards"),
            dones=read("dones"),
            time_outs=read("time_outs"),
            global_env_ids=read("round_global_env_ids"),
            episode_ids=read("round_episode_ids"),
            episode_steps=read("round_episode_steps"),
            observation_is_raytraced=read("round_observation_is_raytraced"),
            infos=MappingProxyType(dict(infos)),
        )


class StepSubscriber(Protocol):
    """Ordered synchronous observer for the validated rollout loop."""

    def on_round_start(self, context: RoundContext) -> None: ...

    def on_decision(self, record: DecisionRecord) -> None: ...

    def on_transition(self, record: StepRecord) -> None: ...

    def on_round_end(self, context: RoundContext) -> None: ...


class RuntimeSubscriber:
    """Convenience base class with no-op subscriber callbacks."""

    __slots__ = ()

    def on_round_start(self, context: RoundContext) -> None:
        del context

    def on_decision(self, record: DecisionRecord) -> None:
        del record

    def on_transition(self, record: StepRecord) -> None:
        del record

    def on_round_end(self, context: RoundContext) -> None:
        del context


@dataclass(frozen=True)
class MetadataAuditSummary:
    """Scalar-only result of one complete metadata audit."""

    num_envs: int
    expected_steps: int
    decision_count: int
    transition_count: int
    callback_count: int
    callback_order_valid: bool
    sequential_steps_valid: bool
    batch_dimensions_valid: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


def _iter_array_metadata(value: Any):
    """Yield only value-free array descriptors from a projected callback."""

    if isinstance(value, ArrayMetadata):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_array_metadata(item)
    elif isinstance(value, tuple):
        for item in value:
            yield from _iter_array_metadata(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _iter_array_metadata(getattr(value, field.name))


class MetadataAuditSubscriber(RuntimeSubscriber):
    """Validate one bounded shadow rollout using metadata and scalar state only.

    The subscriber retains no callback record, array metadata, tensor, NumPy
    value, or user mapping.  It performs no I/O.  Its sole output is a
    scalar-only summary available after a correctly ordered round end.
    """

    __slots__ = (
        "_phase",
        "_started",
        "_ended",
        "_num_envs",
        "_expected_steps",
        "_decision_count",
        "_transition_count",
        "_callback_count",
    )

    def __init__(self) -> None:
        self._phase = "idle"
        self._started = False
        self._ended = False
        self._num_envs = 0
        self._expected_steps = 0
        self._decision_count = 0
        self._transition_count = 0
        self._callback_count = 0

    def _require_batch(self, value: Any, label: str) -> None:
        arrays = tuple(_iter_array_metadata(value))
        if not arrays:
            raise RuntimeAdapterError(
                f"metadata audit found no array metadata in {label}"
            )
        mismatches = tuple(
            metadata.shape
            for metadata in arrays
            if not metadata.shape or metadata.shape[0] != self._num_envs
        )
        if mismatches:
            raise RuntimeAdapterError(
                f"metadata audit batch mismatch in {label}: "
                f"expected {self._num_envs}, got {mismatches}"
            )

    def on_round_start(self, context: RoundContext) -> None:
        if self._started or self._phase != "idle":
            raise RuntimeAdapterError("metadata audit received duplicate round start")
        if not isinstance(context, RoundContext):
            raise RuntimeAdapterError("metadata audit received malformed round context")
        if context.num_envs <= 0 or context.expected_steps <= 0:
            raise RuntimeAdapterError("metadata audit round dimensions must be positive")
        self._num_envs = int(context.num_envs)
        self._expected_steps = int(context.expected_steps)
        self._started = True
        self._phase = "decision"
        self._callback_count = 1

    def on_decision(self, record: DecisionRecord) -> None:
        if self._phase != "decision" or self._ended:
            raise RuntimeAdapterError("metadata audit callback order expected decision")
        if not isinstance(record, DecisionRecord):
            raise RuntimeAdapterError("metadata audit received malformed decision")
        expected_step = self._transition_count
        if record.rollout_step != expected_step:
            raise RuntimeAdapterError(
                "metadata audit decision step is not sequential: "
                f"expected {expected_step}, got {record.rollout_step}"
            )
        if expected_step >= self._expected_steps:
            raise RuntimeAdapterError("metadata audit received too many decisions")
        self._require_batch(record.observations, "decision observations")
        self._require_batch(record.decision, "decision lineage")
        self._decision_count += 1
        self._callback_count += 1
        self._phase = "transition"

    def on_transition(self, record: StepRecord) -> None:
        if self._phase != "transition" or self._ended:
            raise RuntimeAdapterError("metadata audit callback order expected transition")
        if not isinstance(record, StepRecord):
            raise RuntimeAdapterError("metadata audit received malformed transition")
        expected_step = self._transition_count
        if record.rollout_step != expected_step:
            raise RuntimeAdapterError(
                "metadata audit transition step is not sequential: "
                f"expected {expected_step}, got {record.rollout_step}"
            )
        self._require_batch(record.observations, "transition observations")
        self._require_batch(record.decision, "transition decision lineage")
        # Check every canonical storage field independently.  Grouping these
        # values would let one valid array mask a missing or scalar sibling,
        # while the audit summary still claimed complete batch validation.
        for value, label in (
            (record.rewards, "transition rewards"),
            (record.dones, "transition dones"),
            (record.time_outs, "transition time_outs"),
            (record.global_env_ids, "transition global_env_ids"),
            (record.episode_ids, "transition episode_ids"),
            (record.episode_steps, "transition episode_steps"),
            (
                record.observation_is_raytraced,
                "transition observation_is_raytraced",
            ),
        ):
            self._require_batch(value, label)
        self._transition_count += 1
        self._callback_count += 1
        self._phase = (
            "end" if self._transition_count == self._expected_steps else "decision"
        )

    def on_round_end(self, context: RoundContext) -> None:
        if self._phase != "end" or self._ended:
            raise RuntimeAdapterError("metadata audit round ended out of order")
        if not isinstance(context, RoundContext):
            raise RuntimeAdapterError("metadata audit received malformed round context")
        if (
            context.num_envs != self._num_envs
            or context.expected_steps != self._expected_steps
        ):
            raise RuntimeAdapterError("metadata audit round context changed before end")
        if (
            self._decision_count != self._expected_steps
            or self._transition_count != self._expected_steps
        ):
            raise RuntimeAdapterError("metadata audit callback count is incomplete")
        self._callback_count += 1
        self._ended = True
        self._phase = "complete"

    def summary(self) -> MetadataAuditSummary:
        if not self._ended or self._phase != "complete":
            raise RuntimeAdapterError("metadata audit summary requested before round end")
        return MetadataAuditSummary(
            num_envs=self._num_envs,
            expected_steps=self._expected_steps,
            decision_count=self._decision_count,
            transition_count=self._transition_count,
            callback_count=self._callback_count,
            callback_order_valid=True,
            sequential_steps_valid=True,
            batch_dimensions_valid=True,
        )


def _metadata_projection(value: Any, seen: set[int] | None = None) -> Any:
    """Project a record into immutable metadata without retaining tensor values."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return OpaqueMetadata(type_name=f"cycle:{type(value).__qualname__}")
    seen.add(identity)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _metadata_projection(item, seen) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_projection(item, seen) for item in value)
    if hasattr(value, "shape"):
        return ArrayMetadata(
            type_name=type(value).__qualname__,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(getattr(value, "dtype", "")),
            device=str(getattr(value, "device", "")),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _metadata_projection(getattr(value, field.name), seen)
                for field in fields(value)
            }
        )
    return OpaqueMetadata(type_name=type(value).__qualname__)


def _exact_array_equal(left: Any, right: Any) -> bool:
    """Compare a small lineage tensor exactly without a hard framework import."""

    if left is right:
        return True
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape is None or right_shape is None:
        return False
    if tuple(left_shape) != tuple(right_shape):
        return False
    if str(getattr(left, "dtype", "")) != str(getattr(right, "dtype", "")):
        return False
    torch_module = sys.modules.get("torch")
    if torch_module is not None and isinstance(left, torch_module.Tensor):
        return bool(
            isinstance(right, torch_module.Tensor)
            and torch_module.equal(left, right)
        )
    numpy_module = sys.modules.get("numpy")
    if numpy_module is not None and isinstance(left, numpy_module.ndarray):
        return bool(
            isinstance(right, numpy_module.ndarray)
            and numpy_module.array_equal(left, right)
        )
    return False


def _snapshot_array(value: Any) -> Any:
    """Snapshot a live array without introducing a hard framework import.

    The legacy environment is allowed to rewrite the action tensor after the
    rollout storage has copied it (for example, ``DeltaActionBase`` expands a
    policy delta into the cumulative action in-place).  A shadow audit must
    therefore retain the value that existed at the decision/storage boundary,
    not a shallow reference that follows a later environment mutation.

    Only array types that the exact comparator supports are copied.  Test
    doubles and opaque metadata values remain identity-preserving.
    """

    torch_module = sys.modules.get("torch")
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return value.detach().clone()
    numpy_module = sys.modules.get("numpy")
    if numpy_module is not None and isinstance(value, numpy_module.ndarray):
        return value.copy()
    return value


_SHADOW_LINEAGE_KEYS = (
    "student_actions",
    "gt_actions",
    "actions",
    "expert_mask",
    "round_global_env_ids",
    "round_episode_ids",
    "round_episode_steps",
    "round_observation_is_raytraced",
)


@dataclass(frozen=True)
class _RngSnapshot:
    """Global RNG state that can affect a later actor or beta decision."""

    python_state: object
    numpy_module: Any | None
    numpy_state: Any | None
    torch_module: Any | None
    torch_cpu_state: Any | None
    torch_cuda_states: tuple[Any, ...]

    @classmethod
    def capture(cls) -> "_RngSnapshot":
        numpy_module = sys.modules.get("numpy")
        numpy_state = None
        if numpy_module is not None:
            numpy_state = numpy_module.random.get_state()

        torch_module = sys.modules.get("torch")
        torch_cpu_state = None
        torch_cuda_states: tuple[Any, ...] = ()
        if torch_module is not None:
            torch_cpu_state = torch_module.random.get_rng_state()
            # Querying CUDA state initializes CUDA when it has not yet been
            # used.  Avoid changing process state merely to guard a callback.
            if torch_module.cuda.is_initialized():
                torch_cuda_states = tuple(torch_module.cuda.get_rng_state_all())
        return cls(
            python_state=random.getstate(),
            numpy_module=numpy_module,
            numpy_state=numpy_state,
            torch_module=torch_module,
            torch_cpu_state=torch_cpu_state,
            torch_cuda_states=torch_cuda_states,
        )

    @staticmethod
    def _numpy_equal(module: Any, left: Any, right: Any) -> bool:
        return bool(
            left[0] == right[0]
            and module.array_equal(left[1], right[1])
            and left[2:] == right[2:]
        )

    def matches_current(self) -> bool:
        if random.getstate() != self.python_state:
            return False
        if self.numpy_module is not None:
            current_numpy = self.numpy_module.random.get_state()
            if not self._numpy_equal(
                self.numpy_module, self.numpy_state, current_numpy
            ):
                return False
        if self.torch_module is not None:
            current_cpu = self.torch_module.random.get_rng_state()
            if not bool(self.torch_module.equal(self.torch_cpu_state, current_cpu)):
                return False
            if self.torch_cuda_states:
                current_cuda = tuple(self.torch_module.cuda.get_rng_state_all())
                if len(current_cuda) != len(self.torch_cuda_states):
                    return False
                if any(
                    not bool(self.torch_module.equal(expected, current))
                    for expected, current in zip(
                        self.torch_cuda_states, current_cuda
                    )
                ):
                    return False
        return True

    def restore(self) -> None:
        random.setstate(self.python_state)
        if self.numpy_module is not None:
            self.numpy_module.random.set_state(self.numpy_state)
        if self.torch_module is not None:
            self.torch_module.random.set_rng_state(self.torch_cpu_state)
            if self.torch_cuda_states:
                self.torch_module.cuda.set_rng_state_all(
                    list(self.torch_cuda_states)
                )


class SubscriberBus:
    """Ordered, same-thread, fail-closed subscriber delivery.

    Subscribers receive value-free :class:`ArrayMetadata`.  This is an
    in-process extension point for audited callbacks, not a code sandbox.
    """

    def __init__(
        self,
        subscribers: Sequence[RuntimeSubscriber] = (),
        *,
        guard_rng: bool = True,
    ):
        self._subscribers: list[RuntimeSubscriber] = []
        for subscriber in subscribers:
            self._validate_subscriber(subscriber)
            self._subscribers.append(subscriber)
        self._guard_rng = bool(guard_rng)
        self._owner_thread_id: int | None = None
        self._sealed = False

    @property
    def subscribers(self) -> tuple[RuntimeSubscriber, ...]:
        return tuple(self._subscribers)

    @staticmethod
    def _validate_subscriber(subscriber: RuntimeSubscriber) -> None:
        if not isinstance(subscriber, RuntimeSubscriber):
            raise RuntimeAdapterError(
                "runtime subscribers must explicitly subclass RuntimeSubscriber"
            )

    def subscribe(self, subscriber: RuntimeSubscriber) -> None:
        if self._sealed:
            raise RuntimeAdapterError("subscribers cannot be added after dispatch begins")
        self._validate_subscriber(subscriber)
        self._subscribers.append(subscriber)

    def _dispatch(self, callback_name: str, payload: Any) -> None:
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeAdapterError(
                "runtime subscribers must execute on the actor rollout thread"
            )
        self._sealed = True
        if not self._subscribers:
            return
        callback_payload = _metadata_projection(payload)
        for subscriber in self._subscribers:
            callback = getattr(subscriber, callback_name, None)
            if callback is None:
                continue
            rng_before = _RngSnapshot.capture() if self._guard_rng else None
            failure: BaseException | None = None
            try:
                callback(callback_payload)
            except BaseException as exc:  # fail closed, including cancellation
                failure = exc
            rng_changed = (
                rng_before is not None and not rng_before.matches_current()
            )
            if rng_changed:
                rng_before.restore()
            if rng_changed:
                raise SubscriberSideEffectError(
                    f"{type(subscriber).__name__}.{callback_name} changed actor RNG state"
                ) from failure
            if failure is not None:
                raise SubscriberExecutionError(
                    f"{type(subscriber).__name__}.{callback_name} failed"
                ) from failure

    def publish_round_start(self, context: RoundContext) -> None:
        self._dispatch("on_round_start", context)

    def publish_decision(self, record: DecisionRecord) -> None:
        self._dispatch("on_decision", record)

    def publish_transition(self, record: StepRecord) -> None:
        self._dispatch("on_transition", record)

    def publish_round_end(self, context: RoundContext) -> None:
        self._dispatch("on_round_end", context)


class ShadowRuntimeBridge:
    """Stateful phase-one observer attached to the unchanged legacy loop."""

    def __init__(self, subscribers: Sequence[RuntimeSubscriber] = ()):
        self.bus = SubscriberBus(subscribers)
        self._context: RoundContext | None = None
        self._pending: DecisionRecord | None = None
        self._transition_count = 0

    @property
    def has_pending_decision(self) -> bool:
        return self._pending is not None

    @property
    def round_active(self) -> bool:
        return self._context is not None

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def start_round(self, context: RoundContext) -> None:
        if self._context is not None or self._pending is not None:
            raise RuntimeAdapterError("shadow runtime round is already active")
        self._context = context
        self._transition_count = 0
        try:
            self.bus.publish_round_start(context)
        except BaseException:
            self._context = None
            self._transition_count = 0
            raise

    def begin_decision(
        self,
        *,
        rollout_step: int,
        observations: Mapping[str, Any],
        policy_state: Mapping[str, Any],
        beta: float,
    ) -> DecisionRecord:
        if self._context is None:
            raise RuntimeAdapterError("shadow decision arrived outside an active round")
        if self._pending is not None:
            raise RuntimeAdapterError("previous shadow decision has no committed transition")
        step = int(rollout_step)
        if step != self._transition_count:
            raise RuntimeAdapterError(
                f"shadow decision step {step} does not follow {self._transition_count} transitions"
            )
        # RolloutStorage copies these values before ``env.step``.  Some legacy
        # environments then mutate the live action tensor in-place while
        # converting policy deltas to effective actuator commands.  Preserve
        # the exact pre-step values here so the later parity check compares the
        # same semantic boundary as committed storage.  Only one transition is
        # pending at a time, bounding the extra shadow-only memory.
        observation_snapshot = MappingProxyType(
            {key: _snapshot_array(value) for key, value in observations.items()}
        )
        policy_state_snapshot = dict(policy_state)
        for key in _SHADOW_LINEAGE_KEYS:
            if key in policy_state_snapshot:
                policy_state_snapshot[key] = _snapshot_array(
                    policy_state_snapshot[key]
                )
        record = DecisionRecord(
            rollout_step=step,
            observations=observation_snapshot,
            decision=ActionDecision.from_policy_state(
                policy_state_snapshot,
                beta=beta,
            ),
        )
        self.bus.publish_decision(record)
        self._pending = record
        return record

    def complete_transition(
        self,
        storage: Any,
        *,
        observation_keys: Sequence[str],
        infos: Mapping[str, Any],
    ) -> StepRecord:
        if self._context is None or self._pending is None:
            raise RuntimeAdapterError("shadow transition has no pending decision")
        committed_step = int(storage.step) - 1
        if committed_step != self._pending.rollout_step:
            raise RuntimeAdapterError(
                "shadow transition/storage order mismatch: "
                f"decision {self._pending.rollout_step}, storage {committed_step}"
            )
        record = StepRecord.from_storage(
            storage,
            rollout_step=committed_step,
            observation_keys=observation_keys,
            beta=self._pending.decision.beta,
            infos=infos,
        )
        observation_mismatches = tuple(
            key
            for key in observation_keys
            if key not in self._pending.observations
            or not _exact_array_equal(
                self._pending.observations[key], record.observations[key]
            )
        )
        if observation_mismatches:
            raise RuntimeAdapterError(
                "live DAgger observation differs from committed storage: "
                + ", ".join(observation_mismatches)
            )
        pending_state = self._pending.decision.policy_state
        parity_pairs = {
            "student_actions": (
                self._pending.decision.student_actions,
                record.decision.student_actions,
            ),
            "gt_actions": (
                self._pending.decision.expert_actions,
                record.decision.expert_actions,
            ),
            "actions": (
                self._pending.decision.behavior_actions,
                record.decision.behavior_actions,
            ),
            "expert_mask": (
                self._pending.decision.expert_mask,
                record.decision.expert_mask,
            ),
            "round_global_env_ids": (
                pending_state.get("round_global_env_ids"),
                record.global_env_ids,
            ),
            "round_episode_ids": (
                pending_state.get("round_episode_ids"),
                record.episode_ids,
            ),
            "round_episode_steps": (
                pending_state.get("round_episode_steps"),
                record.episode_steps,
            ),
            "round_observation_is_raytraced": (
                pending_state.get("round_observation_is_raytraced"),
                record.observation_is_raytraced,
            ),
        }
        mismatches = tuple(
            name
            for name, (live, committed) in parity_pairs.items()
            if not _exact_array_equal(live, committed)
        )
        if mismatches:
            raise RuntimeAdapterError(
                "live DAgger decision differs from committed storage: "
                + ", ".join(mismatches)
            )
        self.bus.publish_transition(record)
        self._pending = None
        self._transition_count += 1
        return record

    def end_round(self) -> None:
        if self._context is None:
            raise RuntimeAdapterError("shadow runtime round is not active")
        if self._pending is not None:
            raise RuntimeAdapterError("shadow runtime round ended with a pending decision")
        if self._transition_count != self._context.expected_steps:
            raise RuntimeAdapterError(
                "shadow runtime transition count mismatch: "
                f"expected {self._context.expected_steps}, got {self._transition_count}"
            )
        context = self._context
        self.bus.publish_round_end(context)
        self._context = None

    def abort_round(self) -> None:
        """Discard observer state without changing the legacy failure."""

        self._pending = None
        self._context = None
        self._transition_count = 0


__all__ = [
    "ActionDecision",
    "ArrayMetadata",
    "DaggerAgent",
    "DaggerAgentAdapter",
    "DecisionRecord",
    "EnvironmentStep",
    "MetadataAuditSubscriber",
    "MetadataAuditSummary",
    "ObservationTransform",
    "OpaqueMetadata",
    "RUNTIME_ADAPTER_MODE_ENV",
    "RUNTIME_ADAPTER_EVIDENCE_ENV",
    "RUNTIME_ADAPTER_SHADOW_ACK",
    "RUNTIME_ADAPTER_SHADOW_ACK_ENV",
    "RUNTIME_ADAPTER_VALIDATION_ACK",
    "RUNTIME_ADAPTER_VALIDATION_ACK_ENV",
    "RoboRenderObservationTransform",
    "RoundContext",
    "RuntimeAdapterError",
    "RuntimeSubscriber",
    "ShadowRuntimeBridge",
    "StepRecord",
    "StepSubscriber",
    "SubscriberBus",
    "SubscriberExecutionError",
    "SubscriberSideEffectError",
    "VectorEnvironment",
    "VectorEnvironmentAdapter",
    "runtime_adapter_mode",
    "runtime_adapter_validation_evidence_path",
]
