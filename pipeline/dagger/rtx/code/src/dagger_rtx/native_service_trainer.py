"""Certified Raven trainer insertion for the RTX DAgger transition owner.

Hydra constructs this class only after Isaac, the privileged teacher, Homie,
and all Raven camera methods exist.  ``train`` is replaced with a bounded
transaction service; the inherited optimizer/training loop is never entered.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

from gr00t.rl.trl.trainer.distill_trainer_videogen import (  # type: ignore[import-not-found]
    TRLDistillTrainerVideoGen,
)

from .batch_orchestrator import BatchStepOrchestrator
from .batch_owner import IsaacBatchTransitionOwner
from .binary_transport_v2 import BoundedBearerBatchV2Transport
from .binary_transport_v3 import BoundedBearerBatchV3Transport
from .binary_transport_v4 import BoundedBearerBatchV4Transport
from .capacity_telemetry import AppendOnlyTelemetry, RUN_TELEMETRY_SCHEMA
from .contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    ContractError,
    canonical_json,
    payload_digest,
    require_sha256,
)
from .durable import atomic_write_json
from .http_transport import BoundedBearerJsonTransport
from .gateway_topology import bounded_gateway_ports, require_gateway_count
from .lean_client_v2 import LeanH200BatchClient
from .lean_client_v3 import LeanH200BatchClientV3, ShardedLeanH200BatchClientV3
from .lean_client_v4 import LeanH200BatchClientV4, ShardedLeanH200BatchClientV4
from .native_entrypoints import NativeRavenRuntime, build_origin_attested_entrypoints
from .native_raven import build_native_raven_batch
from .phase_telemetry import PhaseBoundaryTelemetry
from .pipelined_window_v4 import (
    AggregateHotPathJournalV4,
    PipelinedBatchStepOrchestratorV4,
    PipelinedBatchWindowRunnerV4,
    TransportTimingObserverV4,
    prewarm_readiness_v4,
)
from .provenance import verify_assets
from .production_control import CoordinatorHTTPControl
from .production_driver import (
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_VALIDATION_INTERVAL,
    PRODUCTION_WINDOW_COUNT,
    RTX_PRO_6000_GPU1_UUID,
    ProductionContinuationSegment,
    ProductionMultiWindowDriver,
    ProductionRunSpec,
    resolve_production_simulator_seed,
    verify_continuation_history,
    verify_episode_namespace_capacity,
    verify_h200_activity_evidence_file,
    verify_predecessor_state_file,
)
from .wandb_phase_mirror import (
    active_raven_wandb_mirror,
    authorize_collection_wandb,
    production_wandb_identity,
)
from .window_runner import (
    BatchWindowRunner,
    RolloutLifecycle,
    verify_ready_with_transient_retry,
)


SERVICE_MODES = {
    "capture-only",
    "isolated-one-step",
    "single-window",
    "multi-window-check",
    "beta-production",
    "capacity-probe",
    "production",
}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required environment variable {name} is unset")
    return value


class NativeDaggerRTXServiceTrainer(TRLDistillTrainerVideoGen):
    """Native vector-environment composition used by the certified launcher."""

    def _rollout_lifecycle(self) -> RolloutLifecycle:
        policy = self.policy_model
        teacher = getattr(self, "ref_model", None)

        def begin() -> None:
            # This mirrors TRLPPOTrainer._rollout_step plus the distillation
            # wrapper: one initialization per physical 81-step window.
            policy.init_rollout()
            if teacher is not None and hasattr(teacher, "init_rollout"):
                try:
                    teacher.init_rollout()
                except Exception as error:
                    # Raven's distillation wrapper treats teacher context
                    # hooks as optional; student/environment failures are not.
                    logging.getLogger(__name__).warning(
                        "Optional Raven teacher init_rollout failed: %s", error
                    )

        def finish() -> None:
            if hasattr(policy, "clear_rollout"):
                policy.clear_rollout()
            if teacher is not None and hasattr(teacher, "clear_rollout"):
                try:
                    teacher.clear_rollout()
                except Exception as error:
                    # Original Actor.clear_rollout deletes distribution, but
                    # deterministic act_inference never recreates it. Raven's
                    # wrapper permits subsequent clears after buffers reset.
                    # Do not edit the teacher, invent a distribution or reset
                    # the simulator to work around this optional hook.
                    logging.getLogger(__name__).warning(
                        "Optional Raven teacher clear_rollout failed: %s", error
                    )

        return RolloutLifecycle(begin=begin, finish=finish)

    def _capture_only(
        self,
        batch: Any,
        runtime: NativeRavenRuntime,
        output: Path,
    ) -> None:
        from .contracts import StepKey

        lifecycle = self._rollout_lifecycle()
        lifecycle.begin()
        try:
            keys = [
                StepKey(
                    "native-capture-smoke",
                    1,
                    f"capture-lane-{lane:03d}",
                    "window-000000",
                    0,
                    f"episode-{lane:03d}-00000000",
                    0,
                )
                for lane in range(batch.num_envs)
            ]
            assignments = batch.bind_episodes({lane: key for lane, key in enumerate(keys)})
            captures = list(batch.capture_batch(keys))
            rows = []
            for lane, capture in enumerate(captures):
                rows.append(
                    {
                        "lane_index": lane,
                        "step_key": capture.key.as_dict(),
                        "observation_source": capture.metadata["observation_source"],
                        "assignment": assignments[lane],
                        "capture_digest": payload_digest(capture.metadata, capture.arrays),
                        "array_shapes": {
                            name: {"shape": list(value.shape), "dtype": value.dtype.name}
                            for name, value in sorted(capture.arrays.items())
                        },
                    }
                )
            atomic_write_json(
                output,
                {
                    "schema": "roborender.dagger.rtx.native-capture-smoke.v1",
                    "result": "PASS",
                    "num_envs": batch.num_envs,
                    "environment_step_calls": 0,
                    "rows": rows,
                },
            )
        finally:
            lifecycle.finish()

    def train(self):  # noqa: D401 - required HuggingFace/Raven trainer hook
        """Enter a bounded native owner mode instead of Raven's training loop."""

        mode = _required_env("DAGGER_RTX_MODE")
        if mode not in SERVICE_MODES:
            raise RuntimeError(f"DAGGER_RTX_MODE must be one of {sorted(SERVICE_MODES)}")
        mse_protocol = None
        mse_beta = None
        native_production_segment = None
        native_segment_sha = None
        heldout_spec = None
        heldout_file = os.environ.get("DAGGER_HELDOUT_COLLECTION_FILE")
        heldout_sha = os.environ.get("DAGGER_HELDOUT_COLLECTION_SHA256")
        if heldout_file or heldout_sha:
            if mode != "single-window" or not (heldout_file and heldout_sha):
                raise RuntimeError("held-out collection requires single-window mode and both sidecar/hash")
            from .heldout_collection_driver import read_actor, verify_registration
            heldout_spec = read_actor(Path(heldout_file), heldout_sha)
            registration = heldout_spec["registration"]
            if (heldout_spec["num_envs"] != int(self.env.num_envs)
                    or registration["run_id"] != _required_env("DAGGER_RUN_ID")
                    or registration["round_id"] != int(_required_env("DAGGER_ROUND_ID"))
                    or heldout_spec["actor_prefix"] != _required_env("DAGGER_ACTOR_PREFIX")
                    or heldout_spec["gpu_uuid"] != os.environ.get("CUDA_VISIBLE_DEVICES")
                    or heldout_spec["physical_gpu_index"] != int(_required_env("DAGGER_PHYSICAL_GPU"))
                    or heldout_spec["gpu_uuid"] != _required_env("DAGGER_PHYSICAL_GPU_UUID")
                    or not callable(getattr(self, "_native_episode_assignment_observer", None))
                    or tuple(heldout_spec["gateway_pids"]) != tuple(map(int, _required_env("DAGGER_EXPECTED_H200_PIDS").split(",")))
                    or os.environ.get("DAGGER_WIRE_PROTOCOL") != "v3" or int(os.environ.get("DAGGER_WINDOW_INDEX", "0")) != 0):
                raise RuntimeError("held-out native launch identity differs")
            heldout_control = CoordinatorHTTPControl("http://127.0.0.1:18610", Path(_required_env("DAGGER_H200_TOKEN_FILE")),
                                                     poll_interval_s=1, wait_timeout_s=90)
            verify_registration(heldout_spec, heldout_control._request)
        mse_file = os.environ.get("DAGGER_MSE_PROTOCOL_FILE")
        mse_file_sha = os.environ.get("DAGGER_MSE_PROTOCOL_SHA256")
        if mse_file or mse_file_sha:
            if mode != "beta-production" or not (mse_file and mse_file_sha):
                raise RuntimeError("MSE opt-in requires beta-production and both DAGGER_MSE_PROTOCOL_FILE/_SHA256")
            from .beta_round import read_config as read_mse_beta_config
            from .mse_round_driver import read_mse_protocol
            mse_beta = read_mse_beta_config(Path(_required_env("DAGGER_BETA_CONFIG")),
                                           _required_env("DAGGER_BETA_CONFIG_SHA256"))
            mse_protocol = read_mse_protocol(Path(mse_file), mse_file_sha, mse_beta)
        segment_file = os.environ.get("DAGGER_PRODUCTION_SEGMENT_FILE")
        native_segment_sha = os.environ.get("DAGGER_PRODUCTION_SEGMENT_SHA256")
        coverage_file = os.environ.get("DAGGER_PRODUCTION_COVERAGE_PLAN_FILE")
        if segment_file or native_segment_sha or coverage_file:
            from .beta_round import is_mse_production
            from .production_segment import read_segment
            if (mse_protocol is None or mode != "beta-production"
                    or not all((segment_file, native_segment_sha, coverage_file))
                    or not is_mse_production(mse_beta)):
                raise RuntimeError("native full-motion production requires the exact MSE segment and staged coverage plan")
            native_production_segment = read_segment(segment_file, native_segment_sha, mse_beta, coverage_file)
            if (not callable(getattr(self, "_native_episode_assignment_observer", None))
                    or not callable(getattr(self, "_native_resource_guard", None))):
                raise RuntimeError("native production reset requires installed assignment and resource observers")
        diagnostic_scope = None
        diagnostic_file = os.environ.get('DAGGER_CONTINUOUS_DIAGNOSTIC_FILE')
        diagnostic_sha = os.environ.get('DAGGER_CONTINUOUS_DIAGNOSTIC_SHA256')
        if diagnostic_file or diagnostic_sha:
            from .continuous_diagnostic import read_scope
            if native_production_segment is None or not (diagnostic_file and diagnostic_sha):
                raise RuntimeError('continuous diagnostic requires its complete immutable native scope')
            diagnostic_scope = read_scope(diagnostic_file, diagnostic_sha,
                config_sha256=_required_env('DAGGER_BETA_CONFIG_SHA256'),
                segment_sha256=native_segment_sha, segment=native_production_segment,
                source_sha256=mse_beta['source_sha256'])
        from .process_memory import compact_host_heap
        compact_host_heap("native_setup_complete")
        runtime_root = Path(_required_env("DAGGER_RUNTIME_ROOT"))
        runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        manifest = Path(_required_env("DAGGER_ASSET_MANIFEST"))
        raven_runtime_root = Path(
            os.environ.get("DAGGER_RAVEN_RUNTIME_ROOT", "/workspace/rtx_BC_transfer")
        )
        static, asset_rows = verify_assets(
            manifest, runtime_raven_root=raven_runtime_root
        )
        # Read the immutable beta config before the first reset/source binding.
        # This is phase-local: a warm owner cannot leak the preceding phase's
        # observation mixture into a new explicitly admitted collection.
        from .observation_contract import configure_native_observations
        observation_config = mse_beta
        if mode == "beta-production" and observation_config is None:
            from .beta_round import read_config
            observation_config = read_config(Path(_required_env("DAGGER_BETA_CONFIG")),
                                            _required_env("DAGGER_BETA_CONFIG_SHA256"))
        configure_native_observations(
            self, None if observation_config is None else observation_config.get("student_observation")
        )
        runtime = NativeRavenRuntime.initialize(self)
        entrypoints = build_origin_attested_entrypoints(self, runtime)
        batch = build_native_raven_batch(self, entrypoints)
        if batch.num_envs != int(self.env.num_envs):
            raise RuntimeError("native batch composition changed env.num_envs")

        if mode == "capture-only":
            if os.environ.get("DAGGER_BENCHMARK_BARRIER_DIR"):
                from .benchmark_barrier import wait_for_benchmark_peer
                wait_for_benchmark_peer(mode, _required_env("DAGGER_ACTOR_PREFIX"))
            self._capture_only(batch, runtime, runtime_root / "capture-smoke.json")
            return {"result": "PASS", "mode": mode, "num_envs": batch.num_envs}

        capacity_mode = mode == "capacity-probe"
        capacity_steps = int(os.environ.get("DAGGER_CAPACITY_STEPS", "0"))
        if capacity_mode and capacity_steps not in {3, 10}:
            raise RuntimeError("capacity-probe requires DAGGER_CAPACITY_STEPS=3 or 10")
        acknowledgements = {
            "isolated-one-step": "I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP",
            "single-window": "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW",
            "multi-window-check": "I_ACKNOWLEDGE_THREE_WINDOWS_AND_REAL_LEARNER_UPDATES",
            "beta-production": "I_ACKNOWLEDGE_BETA_FIFTY_WINDOWS_AND_REAL_LEARNER_UPDATES",
            "capacity-probe": (
                "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS"
                if capacity_steps == 3
                else "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS"
            ),
        }
        if mode == "beta-production":
            from .beta_round import read_config, acknowledgement
            early_beta = read_config(Path(_required_env("DAGGER_BETA_CONFIG")), _required_env("DAGGER_BETA_CONFIG_SHA256"))
            acknowledgements[mode] = acknowledgement(early_beta)
        if mode != "production" and os.environ.get(
            "DAGGER_NATIVE_STEP_ACK"
        ) != acknowledgements[mode]:
            raise RuntimeError(
                "native stepping is fail-closed; DAGGER_NATIVE_STEP_ACK lacks the exact mode acknowledgement"
            )
        token_file = Path(_required_env("DAGGER_H200_TOKEN_FILE"))
        base_url = os.environ.get("DAGGER_H200_BASE_URL", "http://127.0.0.1:18620")
        parsed_base_url = urlsplit(base_url)
        if (
            parsed_base_url.scheme != "http"
            or parsed_base_url.hostname != "127.0.0.1"
            or parsed_base_url.path not in {"", "/"}
            or parsed_base_url.query
            or parsed_base_url.fragment
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
            or parsed_base_url.port is None
            or not 1024 <= parsed_base_url.port <= 65535
        ):
            raise RuntimeError(
                "native service permits only an uncredentialed loopback HTTP H200 boundary"
            )
        run_id = _required_env("DAGGER_RUN_ID")
        round_id = int(_required_env("DAGGER_ROUND_ID"))
        if round_id not in ROUND_BETAS:
            raise RuntimeError("DAGGER_ROUND_ID must be 1..4")
        actor_prefix = _required_env("DAGGER_ACTOR_PREFIX")
        window_index = int(os.environ.get("DAGGER_WINDOW_INDEX", "0"))
        wire_protocol = os.environ.get("DAGGER_WIRE_PROTOCOL", "v1")
        if wire_protocol not in {"v1", "v2", "v3", "v4"}:
            raise RuntimeError("DAGGER_WIRE_PROTOCOL must be v1, v2, v3, or v4")
        if wire_protocol == "v4" and mode != "production":
            raise RuntimeError("v4 is an explicit immutable-production-config opt-in")
        generic_rpc_workers = os.environ.get("DAGGER_H200_RPC_WORKERS")
        legacy_capacity_rpc_workers = os.environ.get("DAGGER_CAPACITY_RPC_WORKERS")
        if generic_rpc_workers is not None and legacy_capacity_rpc_workers is not None:
            if generic_rpc_workers != legacy_capacity_rpc_workers:
                raise RuntimeError(
                    "DAGGER_H200_RPC_WORKERS conflicts with legacy "
                    "DAGGER_CAPACITY_RPC_WORKERS"
                )
        raw_rpc_workers = (
            generic_rpc_workers
            if generic_rpc_workers is not None
            else legacy_capacity_rpc_workers
        )
        if raw_rpc_workers is None:
            raw_rpc_workers = (
                "8"
                if wire_protocol in {"v3", "v4"}
                or mode in {"single-window", "capacity-probe"}
                else "1"
            )
        try:
            rpc_workers = int(raw_rpc_workers)
        except ValueError as exc:
            raise RuntimeError("DAGGER_H200_RPC_WORKERS must be an integer in [1,8]") from exc
        if not 1 <= rpc_workers <= 8:
            raise RuntimeError("DAGGER_H200_RPC_WORKERS must be in [1,8]")
        require_gateway_count(wire_protocol, mode, rpc_workers)
        if (
            wire_protocol not in {"v3", "v4"}
            and mode not in {"single-window", "capacity-probe"}
            and rpc_workers != 1
        ):
            raise RuntimeError(
                "parallel H200 RPC fanout is restricted to single-window or capacity-probe"
            )
        if mode == "production" and wire_protocol not in {"v3", "v4"}:
            raise RuntimeError("production requires a served lean v3 or v4 boundary")
        qualification_config = None
        beta_config = None
        if mode == "beta-production":
            from .beta_round import gpu_uuids, read_config, environments_per_actor, is_mse_smoke, is_mse_production
            from .beta_round import gateway_pids_for_actor, gateway_slots, is_eight_h200
            beta_sha = _required_env("DAGGER_BETA_CONFIG_SHA256")
            beta_config = read_config(Path(_required_env("DAGGER_BETA_CONFIG")), beta_sha)
            bound_gpu_uuids = gpu_uuids(beta_config)
            beta_gpu = int(_required_env("DAGGER_PHYSICAL_GPU"))
            expected_start = native_production_segment["start_iteration_index"] if native_production_segment else 0
            if (wire_protocol != "v3" or rpc_workers != 4 or window_index != expected_start
                    or batch.num_envs != environments_per_actor(beta_config) or round_id != beta_config["round_id"]
                    or beta_config["run_id"] != run_id or beta_gpu not in bound_gpu_uuids
                    or actor_prefix != f"{run_id}-g{beta_gpu}"
                    or _required_env("DAGGER_PHYSICAL_GPU_UUID") != bound_gpu_uuids[beta_gpu]
                    or tuple(map(int, _required_env("DAGGER_EXPECTED_H200_PIDS").split(",")))
                    != tuple(gateway_pids_for_actor(beta_config, beta_gpu))):
                raise RuntimeError("beta production topology/config identity mismatch")
            if is_eight_h200(beta_config) and _required_env('DAGGER_H200_GATEWAY_URLS') != ','.join(
                    f'http://127.0.0.1:{18620+i}' for i in gateway_slots(beta_config,beta_gpu)):
                raise RuntimeError('continuous v2 exclusive RTX/H200 gateway route differs')
            if is_mse_smoke(beta_config) and (mse_protocol is None or not callable(getattr(self, "_native_episode_assignment_observer", None))):
                raise RuntimeError("explicit-motion MSE smoke needs its selection protocol and native assignment observer")
            if is_mse_production(beta_config) and (native_production_segment is None
                    or not callable(getattr(self, "_native_episode_assignment_observer", None))
                    or not callable(getattr(self, "_native_resource_guard", None))):
                raise RuntimeError("full-motion production needs its native segment, reset observer and live resource guard")
        if mode == "multi-window-check":
            from .qualification_driver import GPU_UUIDS, read_config
            qualification_sha = _required_env("DAGGER_QUALIFICATION_CONFIG_SHA256")
            qualification_config = read_config(Path(_required_env("DAGGER_QUALIFICATION_CONFIG")), qualification_sha)
            qualification_gpu = int(_required_env("DAGGER_PHYSICAL_GPU"))
            if (wire_protocol != "v3" or rpc_workers != len(qualification_config["gateway_pids"]) or window_index != 0
                    or batch.num_envs != qualification_config["num_envs_per_gpu"] or round_id != 1
                    or qualification_config["run_id"] != run_id
                    or qualification_gpu not in qualification_config["physical_rtx_gpus"]
                    or _required_env("DAGGER_PHYSICAL_GPU_UUID") != GPU_UUIDS[qualification_gpu]
                    or tuple(map(int, _required_env("DAGGER_EXPECTED_H200_PIDS").split(","))) != tuple(qualification_config["gateway_pids"])):
                raise RuntimeError("bounded multi-window qualification identity mismatch")
        production_spec = None
        production_continuation = None
        production_stop_at_learner_barrier = False
        production_gateway_ports: tuple[int, ...] | None = None
        production_gateway_instance_ids: tuple[str, ...] | None = None
        production_logging: dict[str, Any] | None = None
        production_total_windows = PRODUCTION_WINDOW_COUNT
        production_segment_end_window = PRODUCTION_VALIDATION_INTERVAL
        production_validation_iterations: tuple[int, ...] = (40, 50)
        v4_sharding_strategy = "modulo"
        v4_readiness_prewarm_probes = 0
        v4_suppress_success_transport_events = False
        if mode == "production":
            production_config_path = Path(_required_env("DAGGER_PRODUCTION_CONFIG"))
            production_config_bytes = production_config_path.read_bytes()
            production_config_sha = hashlib.sha256(production_config_bytes).hexdigest()
            production_config = json.loads(production_config_bytes)
            actor = production_config.get("rtx_actor", {})
            h200 = production_config.get("h200", {})
            student = production_config.get("student", {})
            execution = production_config.get("execution", {})
            validation = production_config.get("validation", {})
            production_logging = production_config.get("logging")
            if wire_protocol == "v4":
                configured_sharding = h200.get("sharding_strategy", "modulo")
                configured_prewarm = h200.get("readiness_prewarm_probes", 0)
                configured_suppression = h200.get(
                    "suppress_success_transport_events", False
                )
                if configured_sharding not in {
                    "modulo",
                    "cost_aware_window_pinned",
                }:
                    raise RuntimeError(
                        "h200.sharding_strategy must be modulo or "
                        "cost_aware_window_pinned"
                    )
                if (
                    isinstance(configured_prewarm, bool)
                    or not isinstance(configured_prewarm, int)
                    or not 0 <= configured_prewarm <= 16
                ):
                    raise RuntimeError(
                        "h200.readiness_prewarm_probes must be an integer in [0,16]"
                    )
                if not isinstance(configured_suppression, bool):
                    raise RuntimeError(
                        "h200.suppress_success_transport_events must be boolean"
                    )
                v4_sharding_strategy = configured_sharding
                v4_readiness_prewarm_probes = configured_prewarm
                v4_suppress_success_transport_events = configured_suppression
                environment_sharding = os.environ.get("DAGGER_V4_SHARDING_STRATEGY")
                if (
                    environment_sharding is not None
                    and environment_sharding != v4_sharding_strategy
                ):
                    raise RuntimeError(
                        "DAGGER_V4_SHARDING_STRATEGY differs from immutable config"
                    )
                environment_prewarm = os.environ.get(
                    "DAGGER_V4_READINESS_PREWARM_PROBES"
                )
                if environment_prewarm is not None:
                    try:
                        parsed_environment_prewarm = int(environment_prewarm)
                    except ValueError as exc:
                        raise RuntimeError(
                            "DAGGER_V4_READINESS_PREWARM_PROBES must be an integer"
                        ) from exc
                    if parsed_environment_prewarm != v4_readiness_prewarm_probes:
                        raise RuntimeError(
                            "DAGGER_V4_READINESS_PREWARM_PROBES differs from immutable config"
                        )
            production_stop_at_learner_barrier = execution == {
                "window_count": 1,
                "stop_after_actor_barriers": True,
                "wait_for_checkpoint_advance": False,
                "learner_submission_authorized": False,
                "validation_authorized": False,
            }
            if production_stop_at_learner_barrier:
                production_total_windows = PRODUCTION_WINDOW_COUNT
                production_validation_iterations = (
                    (40, 50) if round_id == 1 else (50,)
                )
            else:
                expected_execution_keys = {
                    "window_count",
                    "stop_after_actor_barriers",
                    "wait_for_checkpoint_advance",
                    "learner_submission_authorized",
                    "validation_authorized",
                }
                configured_total_windows = execution.get("window_count")
                if (
                    set(execution) != expected_execution_keys
                    or isinstance(configured_total_windows, bool)
                    or not isinstance(configured_total_windows, int)
                    or configured_total_windows != PRODUCTION_WINDOW_COUNT
                    or execution.get("stop_after_actor_barriers") is not False
                    or execution.get("wait_for_checkpoint_advance") is not True
                    or execution.get("learner_submission_authorized") is not True
                    or execution.get("validation_authorized") is not True
                ):
                    raise RuntimeError(
                        "production execution must authorize learning/validation and "
                        "request exactly 50 windows"
                    )
                expected_validation_interval = 40 if round_id == 1 else 50
                if validation.get("interval_iterations") != expected_validation_interval:
                    raise RuntimeError(
                        "production validation cadence differs from round schedule"
                    )
                production_total_windows = configured_total_windows
                production_validation_iterations = (
                    (40, 50) if round_id == 1 else (50,)
                )
            if production_stop_at_learner_barrier != (
                os.environ.get("DAGGER_STOP_AT_LEARNER_BARRIER", "0") == "1"
            ):
                raise RuntimeError(
                    "bounded learner-barrier launch flag/config mismatch"
                )
            production_gateway_ports = tuple(h200.get("gateway_ports", ()))
            if (
                len(production_gateway_ports) != 8
                or len(set(production_gateway_ports)) != 8
                or any(
                    isinstance(port, bool)
                    or not isinstance(port, int)
                    or not 1024 <= port <= 65535
                    for port in production_gateway_ports
                )
            ):
                raise RuntimeError(
                    "production requires eight unique loopback gateway ports"
                )
            configured_gateway_instances = tuple(
                h200.get("expected_gateway_instance_ids", ())
            )
            if wire_protocol == "v4":
                if (
                    len(configured_gateway_instances) != 8
                    or len(set(configured_gateway_instances)) != 8
                    or any(
                        not isinstance(instance, str) or len(instance) != 64
                        for instance in configured_gateway_instances
                    )
                ):
                    raise RuntimeError(
                        "v4 production requires eight unique gateway instance identities"
                    )
                production_gateway_instance_ids = configured_gateway_instances
            else:
                production_gateway_instance_ids = None
            coordinator_config_sha = production_config.get(
                "existing_run_config_sha256", production_config_sha
            )
            require_sha256(coordinator_config_sha, "coordinator_config_sha256")
            continuation_path_raw = os.environ.get(
                "DAGGER_PRODUCTION_CONTINUATION", ""
            )
            launched_simulator_seed = int(
                _required_env("DAGGER_SIMULATOR_SEED")
            )
            if continuation_path_raw:
                continuation_path = Path(continuation_path_raw)
                continuation_bytes = continuation_path.read_bytes()
                continuation_sha = hashlib.sha256(continuation_bytes).hexdigest()
                if continuation_sha != _required_env(
                    "DAGGER_PRODUCTION_CONTINUATION_SHA256"
                ):
                    raise RuntimeError(
                        "production continuation manifest SHA-256 mismatch"
                    )
                production_continuation = ProductionContinuationSegment.from_mapping(
                    json.loads(continuation_bytes)
                )
                if (
                    production_continuation.run_id != run_id
                    or production_continuation.config_sha256 != coordinator_config_sha
                    or production_continuation.start_global_window != window_index
                ):
                    raise RuntimeError(
                        "production continuation manifest/run/window identity mismatch"
                    )
            elif window_index != 0:
                raise RuntimeError(
                    "nonzero production window requires an immutable continuation manifest"
                )
            try:
                resolve_production_simulator_seed(
                    actor.get("simulator_seed"),
                    launched_simulator_seed,
                    production_continuation,
                    window_index,
                )
            except (ContractError, ValueError) as exc:
                raise RuntimeError("production simulator seed gate failed") from exc

            initial_checkpoint_sha = require_sha256(
                student.get("initial_checkpoint_sha256"),
                "student.initial_checkpoint_sha256",
            )
            if (
                production_config.get("schema") != "roborender.cross_gpu_dagger_round.v1"
                or production_config.get("run_id") != run_id
                or production_config.get("round_id") != round_id
                or production_config.get("teacher_beta") != ROUND_BETAS[round_id]
                or production_config_sha != _required_env("DAGGER_PRODUCTION_CONFIG_SHA256")
                or actor.get("physical_gpu_index") != 1
                or actor.get("gpu_uuid") != RTX_PRO_6000_GPU1_UUID
                or 0 not in actor.get("forbidden_gpu_indices", [])
                or actor.get("concurrent_environments") != PRODUCTION_BATCH_SIZE
                or actor.get("rollout_steps_per_iteration") != 81
                or actor.get("transitions_per_iteration") != PRODUCTION_BATCH_SIZE * 81
                or actor.get("simulator_seed") != production_config.get("seed")
                or h200.get("render_student_worker_count") != 8
                or h200.get("boundary_contract")
                != f"roborender.dagger.rtx.{wire_protocol}"
                or batch.num_envs != PRODUCTION_BATCH_SIZE
                or actor_prefix != f"{run_id}-g1"
                or os.environ.get("DAGGER_PHYSICAL_GPU") != "1"
                or os.environ.get("DAGGER_PHYSICAL_GPU_UUID") != RTX_PRO_6000_GPU1_UUID
                or os.environ.get("CUDA_VISIBLE_DEVICES") != RTX_PRO_6000_GPU1_UUID
            ):
                raise RuntimeError("production config/GPU/run identity gate failed")
            production_spec = ProductionRunSpec(
                run_id=run_id,
                config_sha256=coordinator_config_sha,
                actor_prefix=actor_prefix,
                actor_ids=tuple(
                    f"{actor_prefix}-lane-{lane:03d}"
                    for lane in range(PRODUCTION_BATCH_SIZE)
                ),
                initial_checkpoint_sha256=student["initial_checkpoint_sha256"],
                round_id=round_id,
                beta=ROUND_BETAS[round_id],
                wire_contract_version=f"roborender.dagger.rtx.{wire_protocol}",
                total_windows=production_total_windows,
                validation_iterations=production_validation_iterations,
            )
            if production_stop_at_learner_barrier:
                if (
                    window_index != 0
                    or production_config.get("seed") != actor.get("simulator_seed")
                    or wire_protocol != "v4"
                ):
                    raise RuntimeError(
                        "bounded v4 learner-barrier config gate failed"
                    )
            if production_continuation is not None:
                verify_continuation_history(
                    production_continuation,
                    continuation_path,
                    continuation_path.parent,
                )
                predecessor_file = Path(
                    _required_env("DAGGER_PRODUCTION_PREDECESSOR_STATE")
                )
                verify_predecessor_state_file(
                    production_continuation,
                    predecessor_file,
                    origin_path=_required_env(
                        "DAGGER_PRODUCTION_PREDECESSOR_STATE_ORIGIN"
                    ),
                )
                activity_file_raw = os.environ.get(
                    "DAGGER_PRODUCTION_H200_ACTIVITY_EVIDENCE", ""
                )
                activity_origin_raw = os.environ.get(
                    "DAGGER_PRODUCTION_H200_ACTIVITY_EVIDENCE_ORIGIN", ""
                )
                verify_h200_activity_evidence_file(
                    production_continuation,
                    Path(activity_file_raw) if activity_file_raw else None,
                    origin_path=activity_origin_raw or None,
                )
                verify_episode_namespace_capacity(
                    production_continuation,
                    (int(value) for value in runtime.seqs.tolist()),
                )
            production_segment_end_window = (
                1
                if production_stop_at_learner_barrier
                else min(
                    boundary
                    for boundary in production_validation_iterations
                    if boundary > window_index
                )
            )
            if production_segment_end_window <= window_index:
                raise RuntimeError(
                    "production segment starts at or beyond the configured window budget"
                )
            expected_production_ack = (
                "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW_TO_LEARNER_BARRIER"
                if production_stop_at_learner_barrier
                else "I_ACKNOWLEDGE_FORTY_NATIVE_81_STEP_WINDOWS"
                if window_index == 0 and round_id == 1
                else "I_ACKNOWLEDGE_FIFTY_NATIVE_81_STEP_WINDOWS_TO_VALIDATION"
                if window_index == 0
                else "I_ACKNOWLEDGE_FRESH_PRODUCTION_CONTINUATION_TO_WINDOW_BUDGET"
                if (
                    production_segment_end_window == production_total_windows
                    and production_total_windows % PRODUCTION_VALIDATION_INTERVAL != 0
                )
                else "I_ACKNOWLEDGE_FRESH_PRODUCTION_CONTINUATION_TO_VALIDATION"
            )
            if os.environ.get("DAGGER_NATIVE_STEP_ACK") != expected_production_ack:
                raise RuntimeError(
                    "native stepping is fail-closed; DAGGER_NATIVE_STEP_ACK lacks "
                    "the exact production-segment acknowledgement"
                )
        # A capacity probe is intentionally not resumable.  Reopening a partial
        # native process would lack a certified simulator/recurrent-state restore.
        owner_state = runtime_root / "owner" / "batch_owner_state.json"
        if capacity_mode and owner_state.exists():
            raise RuntimeError("capacity-probe runtime already contains owner state; restart forbidden")
        telemetry_path = (
            runtime_root / "capacity-telemetry.jsonl"
            if capacity_mode
            else runtime_root / "run-telemetry.jsonl"
            if mode in {"single-window", "multi-window-check", "production", "beta-production"}
            else None
        )
        telemetry = (
            AppendOnlyTelemetry(
                telemetry_path,
                **(
                    {"schema": RUN_TELEMETRY_SCHEMA}
                    if mode in {"single-window", "multi-window-check", "production", "beta-production"}
                    else {}
                ),
            )
            if telemetry_path is not None
            else None
        )
        launch_started_ns = int(os.environ.get("DAGGER_LAUNCH_STARTED_NS", "0"))
        if telemetry is not None and launch_started_ns <= 0:
            raise RuntimeError(f"{mode} requires DAGGER_LAUNCH_STARTED_NS")
        if telemetry is not None:
            telemetry.record(
                "native_service_ready",
                run_id=run_id,
                round_id=round_id,
                actor_prefix=actor_prefix,
                window_index=window_index,
                batch_size=batch.num_envs,
                physical_gpu=int(_required_env("DAGGER_PHYSICAL_GPU")),
                rpc_workers=rpc_workers,
                wire_protocol=wire_protocol,
                initialization_duration_ns=time.time_ns() - launch_started_ns,
            )

        production_execution_steps = (
            (production_segment_end_window - window_index) * 81
        )
        production_telemetry_steps = (
            production_total_windows * 81
        )
        total_vector_steps = (
            81 * beta_config["execution"]["window_count"] if mode == "beta-production" else 243 if mode == "multi-window-check" else capacity_steps
            if capacity_mode
            else production_telemetry_steps
            if mode == "production"
            else 81
            if mode == "single-window"
            else 1
        )
        raw_wandb_enabled = os.environ.get("DAGGER_COLLECTION_WANDB_ENABLED", "0")
        collection_wandb_enabled = authorize_collection_wandb(
            mode=mode,
            raw_enabled=raw_wandb_enabled,
            run_id=run_id,
            production_logging=production_logging,
        )
        wandb_mirror = None
        phase_telemetry = None
        batch_transport = None
        h200_ready_identity = None
        v4_transport_timing = None
        v4_readiness_prewarm = {
            "enabled": False,
            "mode": "readiness_only",
            "probes": 0,
            "ledger_mutations": 0,
            "elapsed_ns": 0,
        }
        try:
            if collection_wandb_enabled:
                # train_agent_trl.py has already performed the sole wandb.init.
                if mode == "production":
                    assert production_logging is not None
                    wandb_run_id, wandb_run_name = production_wandb_identity(
                        run_id, production_logging
                    )
                else:
                    wandb_run_id = wandb_run_name = run_id
                wandb_mirror = active_raven_wandb_mirror(
                    run_id,
                    wandb_run_id=wandb_run_id,
                    wandb_run_name=wandb_run_name,
                )
            phase_telemetry = PhaseBoundaryTelemetry(
                runtime_root / "phase-boundaries.jsonl",
                runtime_root / "progress.json",
                run_id=run_id,
                round_id=round_id,
                total_vector_steps=total_vector_steps,
                batch_size=batch.num_envs,
                fsync_events=False,
                event_observer=wandb_mirror.observe if wandb_mirror is not None else None,
            )

            owner = IsaacBatchTransitionOwner(
                batch,
                runtime_root / "owner",
                runtime_root / "staging",
                static_provenance=static,
                persist_chunks=not capacity_mode,
                ephemeral_hot_path=wire_protocol == "v4",
                telemetry=telemetry,
                phase_telemetry=phase_telemetry,
            )
            if wire_protocol == "v2":
                batch_transport = BoundedBearerBatchV2Transport(
                    base_url,
                    token_file,
                    timeout_s=120.0,
                    telemetry=telemetry,
                )
                transport = LeanH200BatchClient(
                    batch_transport,
                    runtime_root / "h200-v2",
                )
                h200_ready_identity = verify_ready_with_transient_retry(transport)
                orchestrator_rpc_workers = 1
            elif wire_protocol in {"v3", "v4"}:
                raw_gateway_urls = _required_env("DAGGER_H200_GATEWAY_URLS")
                gateway_urls = tuple(raw_gateway_urls.split(","))
                gateway_ports = (
                    production_gateway_ports
                    if production_gateway_ports is not None
                    else bounded_gateway_ports(wire_protocol, mode, rpc_workers, parsed_base_url.port)
                )
                expected_gateway_urls = tuple(
                    f"http://127.0.0.1:{port}" for port in gateway_ports
                )
                if gateway_urls != expected_gateway_urls:
                    raise RuntimeError(
                        f"{wire_protocol} gateway URLs must be the ordered loopback "
                        f"ports {gateway_ports}"
                    )
                if wire_protocol == "v3":
                    gateway_transports = [
                        BoundedBearerBatchV3Transport(
                            url,
                            token_file,
                            timeout_s=120.0,
                            telemetry=telemetry,
                        )
                        for url in gateway_urls
                    ]
                    gateway_clients = [
                        LeanH200BatchClientV3(
                            gateway_transport,
                            runtime_root / "h200-v3" / f"gateway-{index:02d}",
                        )
                        for index, gateway_transport in enumerate(gateway_transports)
                    ]
                    batch_transport = (
                        gateway_clients[0] if rpc_workers == 1
                        else ShardedLeanH200BatchClientV3(gateway_clients, lane_offset=(
                            int(_required_env("DAGGER_PHYSICAL_GPU")) * 2
                            if rpc_workers == 4 and batch.num_envs == 2 and (
                                heldout_spec is not None or
                                beta_config is not None and is_mse_smoke(beta_config))
                            else 0))
                    )
                else:
                    # V4 reuses the audited v3 content-addressed anchor cold
                    # path while all per-step traffic uses RDB4/exchange.
                    v4_transport_timing = TransportTimingObserverV4(
                        telemetry,
                        delegate_success_events=(
                            not v4_suppress_success_transport_events
                        ),
                    )
                    anchor_registrars = [
                        LeanH200BatchClientV3(
                            BoundedBearerBatchV3Transport(
                                url,
                                token_file,
                                timeout_s=120.0,
                                telemetry=v4_transport_timing,
                            ),
                            runtime_root / "h200-v4" / f"gateway-{index:02d}" / "anchors",
                        )
                        for index, url in enumerate(gateway_urls)
                    ]
                    gateway_clients = [
                        LeanH200BatchClientV4(
                            BoundedBearerBatchV4Transport(
                                url,
                                token_file,
                                timeout_s=120.0,
                                telemetry=v4_transport_timing,
                            ),
                            anchor_registrar=anchor_registrars[index],
                        )
                        for index, url in enumerate(gateway_urls)
                    ]
                    batch_transport = ShardedLeanH200BatchClientV4(
                        gateway_clients,
                        cost_aware_sharding=(
                            v4_sharding_strategy == "cost_aware_window_pinned"
                        ),
                    )
                transport = batch_transport
                h200_ready_identity = verify_ready_with_transient_retry(transport)
                if rpc_workers == 1:
                    h200_ready_identity = {"gateways": [h200_ready_identity]}
                expected_pids = tuple(
                    int(value)
                    for value in _required_env("DAGGER_EXPECTED_H200_PIDS").split(",")
                )
                observed_pids = tuple(
                    item.get("pid") for item in h200_ready_identity["gateways"]
                )
                if beta_config is not None:
                    from .beta_round import verify_actor_gateways
                    verify_actor_gateways(beta_config,beta_gpu,h200_ready_identity)
                if (
                    len(expected_pids) != rpc_workers
                    or len(set(expected_pids)) != rpc_workers
                    or observed_pids != expected_pids
                ):
                    raise RuntimeError(
                        f"H200 {wire_protocol} gateway PID identities changed after "
                        "launcher attestation"
                    )
                if production_gateway_instance_ids is not None:
                    observed_instances = tuple(
                        item.get("gateway_instance_id")
                        for item in h200_ready_identity["gateways"]
                    )
                    if observed_instances != production_gateway_instance_ids:
                        raise RuntimeError(
                            "H200 v4 gateway instance identities changed after "
                            "launcher attestation"
                        )
                if wire_protocol == "v4":
                    v4_readiness_prewarm = prewarm_readiness_v4(
                        transport,
                        probes=v4_readiness_prewarm_probes,
                        expected_gateway_pids=expected_pids,
                        expected_gateway_instance_ids=(
                            production_gateway_instance_ids or ()
                        ),
                    )
                    if telemetry is not None:
                        telemetry.record(
                            "v4_readiness_prewarm",
                            **v4_readiness_prewarm,
                        )
                orchestrator_rpc_workers = rpc_workers
            else:
                transport = BoundedBearerJsonTransport(
                    base_url,
                    token_file,
                    timeout_s=120.0,
                    telemetry=telemetry,
                )
                orchestrator_rpc_workers = rpc_workers
            initial_episode_serials = (
                [
                    production_continuation.episode_serial_base + int(value)
                    for value in runtime.seqs.tolist()
                ]
                if production_continuation is not None
                else runtime.seqs.tolist()
            )
            if wire_protocol == "v4":
                orchestrator = PipelinedBatchStepOrchestratorV4(
                    owner,
                    transport,
                    num_envs=batch.num_envs,
                    journal=AggregateHotPathJournalV4(
                        runtime_root / "v4-hot-path.jsonl"
                    ),
                    phase_telemetry=phase_telemetry,
                    transport_timing=v4_transport_timing,
                )
                runner = PipelinedBatchWindowRunnerV4(
                    owner,
                    orchestrator,
                    run_id=run_id,
                    round_id=round_id,
                    actor_prefix=actor_prefix,
                    lifecycle=self._rollout_lifecycle(),
                    initial_episode_serials=initial_episode_serials,
                    phase_telemetry=phase_telemetry,
                )
            else:
                connection_options = {}
                if mode in {"multi-window-check", "beta-production"}:
                    from .connection_health import qualification_connection_options
                    connection_options = qualification_connection_options(
                        os.environ, run_id=run_id, gateway_pids=expected_pids,
                        production=mode == "beta-production")
                if callable(getattr(self, "_native_resource_guard", None)):
                    connection_guard = connection_options.get("request_guard")
                    resource_guard = self._native_resource_guard
                    def guarded_native_request():
                        resource_guard()
                        if connection_guard is not None:
                            connection_guard()
                    connection_options["request_guard"] = guarded_native_request
                behavior_sampler = None
                if beta_config is not None and beta_config.get('behavior_selection_protocol') is not None:
                    from .behavior_selection import sample_vector
                    behavior_sampler = lambda count, beta: sample_vector(count, beta, self.accelerator.device)
                orchestrator = BatchStepOrchestrator(
                    owner,
                    transport,
                    runtime_root / "orchestrator",
                    num_envs=batch.num_envs,
                    max_parallel_requests=orchestrator_rpc_workers,
                    telemetry=telemetry,
                    phase_telemetry=phase_telemetry,
                    behavior_sampler=behavior_sampler,
                )
                runner = BatchWindowRunner(
                    owner,
                    orchestrator,
                    run_id=run_id,
                    round_id=round_id,
                    actor_prefix=actor_prefix,
                    lifecycle=self._rollout_lifecycle(),
                    initial_episode_serials=initial_episode_serials,
                    phase_telemetry=phase_telemetry,
                    episode_namespace=(native_production_segment["episode_namespace"]
                                       if native_production_segment else None),
                    **connection_options,
                )
            from .benchmark_barrier import wait_for_benchmark_peer
            wait_for_benchmark_peer(mode, actor_prefix)
            if mode == "isolated-one-step":
                try:
                    # Native initialization can fail after partially allocating
                    # recurrent rollout state, so cleanup protects begin itself.
                    runner.lifecycle.begin()
                    runner.begin_window(window_index)
                    result = runner.run_step(window_index, 0)
                finally:
                    runner.lifecycle.finish()
            elif mode == "single-window":
                if heldout_spec is None:
                    result = runner.run_window(window_index)
                else:
                    from .heldout_collection_driver import HeldoutCollectionDriver
                    result = HeldoutCollectionDriver(runner, heldout_control._request, runtime_root,
                                                     heldout_spec, heldout_sha).run()
            elif mode == "multi-window-check":
                from .qualification_driver import CommitAckLossProbe, QualificationDriver
                control = CoordinatorHTTPControl("http://127.0.0.1:18610", token_file,
                                                 poll_interval_s=1, wait_timeout_s=90,
                                                 request_guard=runner.request_guard)
                if qualification_gpu == 1:
                    runner.orchestrator.h200 = CommitAckLossProbe(transport, runtime_root / "ack-loss-probe.json")
                result = QualificationDriver(runner, control._request, runtime_root,
                                             qualification_config, qualification_sha, qualification_gpu).run()
                if qualification_gpu == 1 and not runner.orchestrator.h200.injected:
                    raise RuntimeError("registered response-loss probe did not execute")
            elif mode == "beta-production":
                from .qualification_driver import BetaRoundDriver
                control = CoordinatorHTTPControl("http://127.0.0.1:18610", token_file,
                                                 poll_interval_s=1, wait_timeout_s=90,
                                                 request_guard=runner.request_guard)
                driver_class, driver_options = BetaRoundDriver, {}
                if mse_protocol is not None:
                    from .mse_round_driver import MseBetaRoundDriver
                    driver_class = MseBetaRoundDriver
                    driver_options = {"mse_protocol": mse_protocol,
                                      "mse_protocol_file_sha256": mse_file_sha}
                if native_production_segment is not None:
                    from .production_segment import ProductionSegmentDriver, is_continuous_segment
                    driver_class = ProductionSegmentDriver
                    driver_options.update(segment=native_production_segment,
                                          segment_file_sha256=native_segment_sha)
                    if is_continuous_segment(native_production_segment):
                        driver_options.update(
                            qualification_gate_file=Path(segment_file).parent / (
                                'continuous-diagnostic-gate.json' if diagnostic_scope is not None
                                else 'continuous-qualification-gate.json'),
                            resource_snapshot=getattr(self, '_native_resource_summary', None),
                            assignment_snapshot=getattr(self, '_native_assignment_snapshot', None),
                            population_snapshot=getattr(self, '_native_population_snapshot', None),
                            reset_inventory_binding=(dict(protocol=beta_config['reset_inventory_protocol'],
                                **self.env.simulator.native_reset_inventory_binding)
                                if beta_config.get('reset_inventory_protocol') is not None else None),
                            diagnostic_scope=diagnostic_scope, diagnostic_scope_sha256=diagnostic_sha,
                            h200_gateway_identities=verify_actor_gateways(beta_config,beta_gpu,h200_ready_identity),
                            coverage_plan_document_sha256=hashlib.sha256(canonical_json(
                                json.loads(Path(coverage_file).read_bytes()))).hexdigest())
                result = driver_class(runner, control._request, runtime_root,
                                      beta_config, beta_sha, beta_gpu, **driver_options).run()
            elif mode == "production":
                if production_spec is None:
                    raise RuntimeError("production spec was not constructed")
                if wire_protocol == "v2":
                    expected_h200_pid = int(_required_env("DAGGER_EXPECTED_H200_PID"))
                    if h200_ready_identity.get("pid") != expected_h200_pid:
                        raise RuntimeError(
                            "H200 render/student PID changed after launcher attestation"
                        )
                else:
                    # Sharded v3/v4 PID order was already attested before the
                    # orchestrator was constructed.
                    pass
                control = CoordinatorHTTPControl(
                    os.environ.get(
                        "DAGGER_H200_COORDINATOR_URL", "http://127.0.0.1:18610"
                    ),
                    token_file,
                )
                result = ProductionMultiWindowDriver(
                    runner,
                    control,
                    runtime_root,
                    production_spec,
                    continuation=production_continuation,
                    stop_at_learner_barrier=production_stop_at_learner_barrier,
                ).run()
            else:
                result = runner.run_bounded_probe(window_index, steps=capacity_steps)
        except BaseException:
            if batch_transport is not None:
                batch_transport.close()
            if wandb_mirror is not None:
                wandb_mirror.finish(exit_code=1)
            if phase_telemetry is not None:
                phase_telemetry.close()
            raise
        if batch_transport is not None:
            batch_transport.close()
        if wandb_mirror is not None:
            wandb_mirror.finish(exit_code=0)
            wandb_summary = wandb_mirror.summary()
        else:
            wandb_summary = {"enabled": False}
        phase_telemetry.close()
        summary = {
            "schema": CONTRACT_VERSION,
            "result": "PASS",
            "mode": mode,
            "num_envs": batch.num_envs,
            "physical_gpu": int(os.environ.get("DAGGER_PHYSICAL_GPU", "0")),
            "rpc_workers": rpc_workers,
            "wire_protocol": wire_protocol,
            "h200_ready_identity": h200_ready_identity,
            "v4_sharding_strategy": (
                v4_sharding_strategy if wire_protocol == "v4" else None
            ),
            "v4_readiness_prewarm": (
                v4_readiness_prewarm if wire_protocol == "v4" else None
            ),
            "v4_transport_success_telemetry": (
                (
                    "aggregate_vector_step_no_fsync"
                    if v4_suppress_success_transport_events
                    else "durable_per_request_and_aggregate_vector_step"
                )
                if wire_protocol == "v4"
                else None
            ),
            "capacity_steps": capacity_steps if capacity_mode else None,
            "chunk_persistence": not capacity_mode,
            "restart_continuation_allowed": (
                False if mode in {"capacity-probe", "production"} else None
            ),
            "telemetry_path": (
                str(telemetry_path) if telemetry_path is not None else None
            ),
            "collection_wandb": wandb_summary,
            "asset_verification": asset_rows,
            "transaction": result,
            "diagnostic_only": diagnostic_scope is not None,
            "production_round_complete": (False if diagnostic_scope is not None else None),
        }
        # Non-secret terminal attestation is readable by the host's detached
        # resource owner before Docker --rm/chown cleanup. Private journals,
        # payloads and control receipts retain the default owner-only mode.
        atomic_write_json(runtime_root / "native-service-result.json", summary, mode=0o644)
        if telemetry is not None:
            complete_detail = {
                "result": "PASS",
                "transition_batches": (
                    capacity_steps
                    if capacity_mode
                    else production_execution_steps
                    if mode == "production"
                    else result["transition_batches"]
                ),
                "physical_transition_rows": (
                    capacity_steps * batch.num_envs
                    if capacity_mode
                    else production_execution_steps * batch.num_envs
                    if mode == "production"
                    else result["physical_transition_rows"]
                ),
                "no_chunk_or_seal": capacity_mode,
            }
            telemetry.record("native_service_complete", **complete_detail)
        return summary
