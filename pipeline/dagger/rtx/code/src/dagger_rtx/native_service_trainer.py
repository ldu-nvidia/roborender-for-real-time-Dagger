"""Certified Raven trainer insertion for the RTX DAgger transition owner.

Hydra constructs this class only after Isaac, the privileged teacher, Homie,
and all Raven camera methods exist.  ``train`` is replaced with a bounded
transaction service; the inherited optimizer/training loop is never entered.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any

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
    canonical_json,
    payload_digest,
    require_sha256,
)
from .durable import atomic_write_json
from .http_transport import BoundedBearerJsonTransport
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
)
from .provenance import verify_assets
from .production_control import CoordinatorHTTPControl
from .production_driver import (
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_WINDOW_COUNT,
    RTX_PRO_6000_GPU1_UUID,
    ProductionContinuationSegment,
    ProductionMultiWindowDriver,
    ProductionRunSpec,
    verify_continuation_history,
    verify_episode_namespace_capacity,
    verify_h200_activity_evidence_file,
    verify_predecessor_state_file,
)
from .wandb_phase_mirror import active_raven_wandb_mirror
from .window_runner import BatchWindowRunner, RolloutLifecycle


SERVICE_MODES = {
    "capture-only",
    "isolated-one-step",
    "single-window",
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
                teacher.init_rollout()

        def finish() -> None:
            try:
                if hasattr(policy, "clear_rollout"):
                    policy.clear_rollout()
            finally:
                if teacher is not None and hasattr(teacher, "clear_rollout"):
                    teacher.clear_rollout()

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
        runtime_root = Path(_required_env("DAGGER_RUNTIME_ROOT"))
        runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        manifest = Path(_required_env("DAGGER_ASSET_MANIFEST"))
        raven_runtime_root = Path(
            os.environ.get("DAGGER_RAVEN_RUNTIME_ROOT", "/workspace/rtx_BC_transfer")
        )
        static, asset_rows = verify_assets(
            manifest, runtime_raven_root=raven_runtime_root
        )
        runtime = NativeRavenRuntime.initialize(self)
        entrypoints = build_origin_attested_entrypoints(self, runtime)
        batch = build_native_raven_batch(self, entrypoints)
        if batch.num_envs != int(self.env.num_envs):
            raise RuntimeError("native batch composition changed env.num_envs")

        if mode == "capture-only":
            self._capture_only(batch, runtime, runtime_root / "capture-smoke.json")
            return {"result": "PASS", "mode": mode, "num_envs": batch.num_envs}

        capacity_mode = mode == "capacity-probe"
        capacity_steps = int(os.environ.get("DAGGER_CAPACITY_STEPS", "0"))
        if capacity_mode and capacity_steps not in {3, 10}:
            raise RuntimeError("capacity-probe requires DAGGER_CAPACITY_STEPS=3 or 10")
        acknowledgements = {
            "isolated-one-step": "I_ACKNOWLEDGE_ONE_NATIVE_VECTOR_STEP",
            "single-window": "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW",
            "capacity-probe": (
                "I_ACKNOWLEDGE_THREE_NATIVE_VECTOR_STEPS"
                if capacity_steps == 3
                else "I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS"
            ),
            "production": (
                "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW_TO_LEARNER_BARRIER"
                if os.environ.get("DAGGER_STOP_AT_LEARNER_BARRIER", "0") == "1"
                else "I_ACKNOWLEDGE_FRESH_PRODUCTION_CONTINUATION_TO_VALIDATION"
                if os.environ.get("DAGGER_PRODUCTION_CONTINUATION", "")
                else "I_ACKNOWLEDGE_FORTY_NATIVE_81_STEP_WINDOWS"
            ),
        }
        acknowledgement = acknowledgements[mode]
        if os.environ.get("DAGGER_NATIVE_STEP_ACK") != acknowledgement:
            raise RuntimeError(
                "native stepping is fail-closed; DAGGER_NATIVE_STEP_ACK lacks the exact mode acknowledgement"
            )
        token_file = Path(_required_env("DAGGER_H200_TOKEN_FILE"))
        base_url = os.environ.get("DAGGER_H200_BASE_URL", "http://127.0.0.1:18620")
        if base_url not in {
            "http://127.0.0.1:18620",
            "http://127.0.0.1:18720",
        }:
            raise RuntimeError(
                "native service permits only an attested loopback H200 boundary"
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
        if wire_protocol in {"v3", "v4"} and rpc_workers != 8:
            raise RuntimeError(
                f"{wire_protocol} requires exactly eight H200 gateway RPC workers"
            )
        if (
            wire_protocol not in {"v3", "v4"}
            and mode not in {"single-window", "capacity-probe"}
            and rpc_workers != 1
        ):
            raise RuntimeError(
                "parallel H200 RPC fanout is restricted to single-window or capacity-probe"
            )
        if mode == "production" and wire_protocol not in {"v2", "v3", "v4"}:
            raise RuntimeError("production requires a lean v2, v3, or v4 boundary")
        production_spec = None
        production_continuation = None
        production_stop_at_learner_barrier = False
        production_gateway_ports: tuple[int, ...] | None = None
        production_gateway_instance_ids: tuple[str, ...] | None = None
        if mode == "production":
            production_config_path = Path(_required_env("DAGGER_PRODUCTION_CONFIG"))
            production_config_bytes = production_config_path.read_bytes()
            production_config_sha = hashlib.sha256(production_config_bytes).hexdigest()
            production_config = json.loads(production_config_bytes)
            actor = production_config.get("rtx_actor", {})
            h200 = production_config.get("h200", {})
            student = production_config.get("student", {})
            execution = production_config.get("execution", {})
            production_stop_at_learner_barrier = execution == {
                "window_count": 1,
                "stop_after_actor_barriers": True,
                "wait_for_checkpoint_advance": False,
                "learner_submission_authorized": False,
                "validation_authorized": False,
            }
            if production_stop_at_learner_barrier != (
                os.environ.get("DAGGER_STOP_AT_LEARNER_BARRIER", "0") == "1"
            ):
                raise RuntimeError(
                    "bounded learner-barrier launch flag/config mismatch"
                )
            if production_stop_at_learner_barrier:
                production_gateway_ports = tuple(h200.get("gateway_ports", ()))
                production_gateway_instance_ids = tuple(
                    h200.get("expected_gateway_instance_ids", ())
                )
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
                or actor.get("simulator_seed")
                != int(_required_env("DAGGER_SIMULATOR_SEED"))
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
                config_sha256=production_config_sha,
                actor_prefix=actor_prefix,
                actor_ids=tuple(
                    f"{actor_prefix}-lane-{lane:03d}"
                    for lane in range(PRODUCTION_BATCH_SIZE)
                ),
                initial_checkpoint_sha256=student["initial_checkpoint_sha256"],
                wire_contract_version=f"roborender.dagger.rtx.{wire_protocol}",
            )
            if production_stop_at_learner_barrier:
                if (
                    window_index != 0
                    or production_config.get("seed") != 4242
                    or actor.get("simulator_seed") != 4242
                    or wire_protocol != "v4"
                    or production_gateway_ports != tuple(range(18720, 18728))
                    or len(production_gateway_instance_ids) != 8
                    or len(set(production_gateway_instance_ids)) != 8
                    or initial_checkpoint_sha
                    != "36138c34d68378b26265a9ff1f0e854ca1f6bc4c4bb5e582a2a7a9301f9fea33"
                ):
                    raise RuntimeError(
                        "bounded v4 learner-barrier smoke identity/config gate failed"
                    )
            continuation_path_raw = os.environ.get(
                "DAGGER_PRODUCTION_CONTINUATION", ""
            )
            if continuation_path_raw:
                continuation_path = Path(continuation_path_raw)
                continuation_bytes = continuation_path.read_bytes()
                continuation_sha = hashlib.sha256(continuation_bytes).hexdigest()
                if continuation_sha != _required_env(
                    "DAGGER_PRODUCTION_CONTINUATION_SHA256"
                ):
                    raise RuntimeError("production continuation manifest SHA-256 mismatch")
                production_continuation = ProductionContinuationSegment.from_mapping(
                    json.loads(continuation_bytes)
                )
                if (
                    production_continuation.run_id != run_id
                    or production_continuation.config_sha256 != production_config_sha
                    or production_continuation.start_global_window != window_index
                ):
                    raise RuntimeError(
                        "production continuation manifest/run/window identity mismatch"
                    )
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
            elif window_index != 0:
                raise RuntimeError(
                    "nonzero production window requires an immutable continuation manifest"
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
            if mode in {"single-window", "production"}
            else None
        )
        telemetry = (
            AppendOnlyTelemetry(
                telemetry_path,
                **(
                    {"schema": RUN_TELEMETRY_SCHEMA}
                    if mode in {"single-window", "production"}
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
            (
                production_continuation.end_global_window_exclusive
                - production_continuation.start_global_window
            )
            * 81
            if production_continuation is not None
            else (81 if production_stop_at_learner_barrier else PRODUCTION_WINDOW_COUNT * 81)
        )
        production_telemetry_steps = (
            production_continuation.end_global_window_exclusive * 81
            if production_continuation is not None
            else (81 if production_stop_at_learner_barrier else PRODUCTION_WINDOW_COUNT * 81)
        )
        total_vector_steps = (
            capacity_steps
            if capacity_mode
            else production_telemetry_steps
            if mode == "production"
            else 81
            if mode == "single-window"
            else 1
        )
        raw_wandb_enabled = os.environ.get("DAGGER_COLLECTION_WANDB_ENABLED", "0")
        if raw_wandb_enabled not in {"0", "1"}:
            raise RuntimeError("DAGGER_COLLECTION_WANDB_ENABLED must be 0 or 1")
        collection_wandb_enabled = raw_wandb_enabled == "1"
        if collection_wandb_enabled and not (
            mode == "single-window"
            or (mode == "production" and production_stop_at_learner_barrier)
        ):
            raise RuntimeError(
                "collection-side W&B is restricted to a bounded one-window run"
            )
        wandb_mirror = None
        phase_telemetry = None
        batch_transport = None
        h200_ready_identity = None
        try:
            if collection_wandb_enabled:
                # train_agent_trl.py has already performed the sole wandb.init.
                wandb_mirror = active_raven_wandb_mirror(run_id)
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
                h200_ready_identity = transport.verify_ready()
                orchestrator_rpc_workers = 1
            elif wire_protocol in {"v3", "v4"}:
                raw_gateway_urls = _required_env("DAGGER_H200_GATEWAY_URLS")
                gateway_urls = tuple(raw_gateway_urls.split(","))
                gateway_ports = (
                    production_gateway_ports
                    if production_gateway_ports is not None
                    else tuple(range(18620, 18628))
                )
                expected_gateway_urls = tuple(
                    f"http://127.0.0.1:{port}" for port in gateway_ports
                )
                if gateway_urls != expected_gateway_urls:
                    raise RuntimeError(
                        f"{wire_protocol} gateway URLs must be the ordered loopback "
                        "ports 18620..18627"
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
                    batch_transport = ShardedLeanH200BatchClientV3(gateway_clients)
                else:
                    # V4 reuses the audited v3 content-addressed anchor cold
                    # path while all per-step traffic uses RDB4/exchange.
                    anchor_registrars = [
                        LeanH200BatchClientV3(
                            BoundedBearerBatchV3Transport(
                                url,
                                token_file,
                                timeout_s=120.0,
                                telemetry=telemetry,
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
                                telemetry=telemetry,
                            ),
                            anchor_registrar=anchor_registrars[index],
                        )
                        for index, url in enumerate(gateway_urls)
                    ]
                    batch_transport = ShardedLeanH200BatchClientV4(gateway_clients)
                transport = batch_transport
                h200_ready_identity = transport.verify_ready()
                expected_pids = tuple(
                    int(value)
                    for value in _required_env("DAGGER_EXPECTED_H200_PIDS").split(",")
                )
                observed_pids = tuple(
                    item.get("pid") for item in h200_ready_identity["gateways"]
                )
                if (
                    len(expected_pids) != 8
                    or len(set(expected_pids)) != 8
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
                orchestrator_rpc_workers = 8
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
                orchestrator = BatchStepOrchestrator(
                    owner,
                    transport,
                    runtime_root / "orchestrator",
                    num_envs=batch.num_envs,
                    max_parallel_requests=orchestrator_rpc_workers,
                    telemetry=telemetry,
                    phase_telemetry=phase_telemetry,
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
                )
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
                result = runner.run_window(window_index)
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
            "capacity_steps": capacity_steps if capacity_mode else None,
            "chunk_persistence": not capacity_mode,
            "restart_continuation_allowed": False if mode in {"capacity-probe", "production"} else None,
            "telemetry_path": str(telemetry_path) if telemetry_path is not None else None,
            "collection_wandb": wandb_summary,
            "asset_verification": asset_rows,
            "transaction": result,
        }
        atomic_write_json(runtime_root / "native-service-result.json", summary)
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
