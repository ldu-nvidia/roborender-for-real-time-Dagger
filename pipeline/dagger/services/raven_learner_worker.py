#!/usr/bin/env python3
"""One Raven-equivalent supervised update over barrier-admitted DAgger chunks."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# The learner service launches this executable by absolute path under Raven's
# hardened PYTHONSAFEPATH runtime.  Add only this file's audited project root;
# do not depend on the caller's working directory or ambient PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.services.contracts import (
    CANONICAL_BETA,
    canonical_json,
    file_sha256,
    validate_dagger_chunk,
)
from pipeline.dagger.services.phase_telemetry import PhaseTelemetry
from pipeline.dagger.services.metric_contract import (
    LEGACY as LEGACY_METRIC_PROTOCOL,
    MSE_FINAL_TEST,
    metric_config,
    pending_candidate_metrics,
)
from pipeline.dagger.services.mse_status import candidate_summary_status
from pipeline.dagger.services import bc32_diagnostic_parent as diagnostic_parent
from pipeline.dagger.services import learner_rng
from pipeline.dagger.services import learner_order
from pipeline.dagger.services import learner_memory
from pipeline.dagger.services import diagnostic_reentry

REPLAY_DAGGER_SCHEMA = 'roborender.dagger_replay_student_checkpoint.v1'
REPLAY_LINEAGE_SCHEMA = 'roborender.raven-replay-optimizer-lineage.v1'


def uses_replay_optimizer(checkpoint):
    batch = checkpoint.get('training_batch_contract') or {}
    if not isinstance(batch,dict):
        batch={}
    return (checkpoint.get('schema') == REPLAY_DAGGER_SCHEMA
        or (checkpoint.get('schema') == 'roborender.bc_student_checkpoint.v4'
            and (batch.get('student_lifecycle') == 'raven_offline_replay_v1'
                or batch.get('schema') in ('raven_offline_replay_16x81_activation_checkpoint_v2',
                    'raven_offline_replay_32x81_activation_checkpoint_v1'))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args()


def build_policy(spec: dict, state: dict, device: torch.device):
    # Keep the service/test control plane importable without the heavyweight
    # GR00T runtime.  The worker imports these only when it constructs a model.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from pipeline.dagger.services.runtime_policy_contract import attest_policy_runtime

    runtime_attestation = attest_policy_runtime()

    env = OmegaConf.create({
        "robot": spec["env_config"]["robot"],
        "simulator": {"config": {"cameras": {
            "camera_resolutions": [108, 192], "camera_types": [{"rgb": True}],
        }}},
    })
    policy = instantiate(
        OmegaConf.create(spec["actor_config"]),
        env_config=env,
        algo_config=OmegaConf.create(spec["algo_config"]),
        module_dim_dict=spec["module_dim_dict"],
        _recursive_=False,
    )
    policy.load_state_dict(state, strict=True)
    policy._roborender_runtime_attestation = runtime_attestation
    for parameter in policy.obj_pred_mlp.parameters():
        parameter.requires_grad_(False)
    if hasattr(policy, "std"):
        policy.std.requires_grad_(False)
    return policy.to(device)


def chunk_paths(state_root: Path, chunk_ids: list[str], *,
                deep_validation_samples: int, sample_seed: str) -> list[Path]:
    from pipeline.dagger.services.ledger_owner import open_readonly
    database = open_readonly(state_root / "state.sqlite3")
    database.row_factory = sqlite3.Row
    try:
        paths = []
        for chunk_id in chunk_ids:
            row = database.execute(
                "SELECT path,status,size_bytes,sha256 FROM chunk_reservations WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if row is None or row["status"] != "committed":
                raise RuntimeError(f"chunk is not committed: {chunk_id}")
            path = Path(row["path"]).resolve()
            if not path.is_file() or path.stat().st_size != row["size_bytes"]:
                raise RuntimeError(f"chunk is missing or has the wrong size: {chunk_id}")
            paths.append(path)
        sample_count = min(max(0, deep_validation_samples), len(paths))
        sampled = sorted(
            zip(chunk_ids, paths),
            key=lambda pair: hashlib.sha256(
                f"{sample_seed}:{pair[0]}".encode()
            ).hexdigest(),
        )[:sample_count]
        for chunk_id, path in sampled:
            row = database.execute(
                "SELECT sha256 FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"sampled chunk hash failed verification: {chunk_id}")
            validate_dagger_chunk(path)
        return paths
    finally:
        database.close()


def replay_training_paths(
    state_root: Path, chunk_ids: list[str], replay_manifest: dict | None, *,
    deep_validation_samples: int, sample_seed: str,
) -> list[Path]:
    """Resolve the immutable replay order recorded in the learner request."""

    dagger_paths = chunk_paths(
        state_root, chunk_ids, deep_validation_samples=deep_validation_samples,
        sample_seed=sample_seed,
    )
    if replay_manifest is None:
        return dagger_paths
    by_id = dict(zip(chunk_ids, dagger_paths))
    paths = []
    for sample in replay_manifest["samples"]:
        if sample["source"] == "bc":
            path = Path(sample["path"]).resolve()
            if not path.is_file():
                raise RuntimeError(f"BC replay chunk is missing: {sample['chunk_id']}")
            paths.append(path)
        else:
            try:
                paths.append(by_id[sample["chunk_id"]])
            except KeyError as exc:
                raise RuntimeError("replay manifest references an unadmitted chunk") from exc
    if len(paths) != replay_manifest["total_chunks"]:
        raise RuntimeError("resolved replay batch size differs from its manifest")
    return paths


def load_batch(paths: list[Path], device: torch.device, spec: dict):
    views = spec["input_contract"]["views"]
    vision, actor, target = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as chunk:
            rgb_views = []
            for view in views:
                rgb = chunk[f"rgb_gen__{view}"]
                # Corrected offline BC archives retain uint8 pixels, while the
                # online H200 ledger stores lossless float32 [0,1].  Convert
                # exactly once so both sources enter the same policy contract.
                if rgb.dtype == np.uint8:
                    rgb = rgb.astype(np.float32) / 255.0
                elif rgb.dtype != np.float32:
                    raise RuntimeError(f"unsupported replay RGB dtype: {rgb.dtype}")
                rgb_views.append(rgb)
            vision.append(np.concatenate(rgb_views, axis=1))
            actor.append(chunk["obs_actor_obs"])
            target.append(chunk["gt_actions"][:, : spec["output_contract"]["policy_action_dim"]])
    # Online H200 chunks store both native ray-traced and Image3F observations
    # in Raven's lossless float32 [0,1] policy-view contract.
    vision_tensor = torch.from_numpy(np.stack(vision)).to(device=device, dtype=torch.float32)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    vision_tensor = (vision_tensor - mean) / std
    return (
        vision_tensor,
        torch.from_numpy(np.stack(actor)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(target)).to(device=device, dtype=torch.float32),
    )


class IndexSafeMaxPool2d(torch.nn.Module):
    """A MaxPool2d with identical math and bounded per-kernel tensor indexing.

    CUDA's legacy max-pool implementation uses signed 32-bit indexing for a
    single invocation.  Raven's three-view ``B*T`` tensor can put the first
    ResNet pool just above that limit even though the model fits in H200
    memory.  Batch samples are independent in max pooling, so splitting only
    this operation on dimension zero and concatenating its outputs is exact;
    importantly, this does *not* split any BatchNorm reduction.
    """

    def __init__(self, pool: torch.nn.MaxPool2d, *, max_input_elements: int = 2**31 - 1):
        super().__init__()
        if max_input_elements < 1:
            raise ValueError("max_input_elements must be positive")
        self.kernel_size = pool.kernel_size
        self.stride = pool.stride
        self.padding = pool.padding
        self.dilation = pool.dilation
        self.return_indices = pool.return_indices
        self.ceil_mode = pool.ceil_mode
        self.max_input_elements = int(max_input_elements)

    def _pool(self, value: torch.Tensor):
        return F.max_pool2d(
            value,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )

    def forward(self, value: torch.Tensor):
        if value.ndim < 1 or value.shape[0] < 1:
            return self._pool(value)
        elements_per_sample = value[0].numel()
        if elements_per_sample < 1 or value.numel() <= self.max_input_elements:
            return self._pool(value)
        samples_per_call = self.max_input_elements // elements_per_sample
        if samples_per_call < 1:
            raise RuntimeError(
                "one max-pool sample exceeds the configured safe indexing limit"
            )
        outputs = [
            self._pool(value[start : start + samples_per_call])
            for start in range(0, value.shape[0], samples_per_call)
        ]
        if self.return_indices:
            pooled, indices = zip(*outputs)
            return torch.cat(pooled, dim=0), torch.cat(indices, dim=0)
        return torch.cat(outputs, dim=0)


def install_index_safe_max_pool2d(
    module: torch.nn.Module, *, max_input_elements: int = 2**31 - 1
) -> int:
    """Replace MaxPool2d descendants without touching BatchNorm boundaries."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, IndexSafeMaxPool2d):
            continue
        if isinstance(child, torch.nn.MaxPool2d):
            setattr(
                module,
                name,
                IndexSafeMaxPool2d(
                    child, max_input_elements=max_input_elements
                ),
            )
            replaced += 1
        else:
            replaced += install_index_safe_max_pool2d(
                child, max_input_elements=max_input_elements
            )
    return replaced


@contextlib.contextmanager
def _batch_norm_recompute_context(module: torch.nn.Module):
    """Restore BatchNorm running buffers after activation recomputation.

    The recompute must execute the identical training graph (including the
    running-stat branch) for PyTorch checkpoint consistency. Snapshotting the
    already-correct post-forward buffers and restoring them afterward prevents
    the recompute from applying momentum or ``num_batches_tracked`` twice.
    """

    batch_norms = [
        child
        for child in module.modules()
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
    ]
    snapshots = [
        (
            child.running_mean.detach().clone() if child.running_mean is not None else None,
            child.running_var.detach().clone() if child.running_var is not None else None,
            child.num_batches_tracked.detach().clone()
            if child.num_batches_tracked is not None
            else None,
        )
        for child in batch_norms
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for child, (running_mean, running_var, batches) in zip(
                batch_norms, snapshots, strict=True
            ):
                if running_mean is not None:
                    child.running_mean.copy_(running_mean)
                if running_var is not None:
                    child.running_var.copy_(running_var)
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)


def checkpointed_full_window_actions(
    policy: torch.nn.Module, vision: torch.Tensor, actor: torch.Tensor
) -> torch.Tensor:
    """Run one exact full-window forward with activation recomputation.

    Non-reentrant checkpointing keeps the full ``B*T`` normalization domain
    but does not retain the complete ResNet activation graph. During backward
    recomputation BatchNorm still uses the same full-window batch statistics;
    only its running-buffer mutation is suppressed so Raven's one-forward
    ``num_batches_tracked``/momentum semantics are preserved exactly.
    """

    def action_forward(vision_value: torch.Tensor, actor_value: torch.Tensor):
        return policy({
            "vision_obs": vision_value,
            "actor_obs": actor_value,
        })["actions"]

    def checkpoint_contexts():
        return contextlib.nullcontext(), _batch_norm_recompute_context(policy)

    return checkpoint(
        action_forward,
        vision,
        actor,
        use_reentrant=False,
        context_fn=checkpoint_contexts,
    )


def backward_global_mse_in_environment_microbatches(
    policy,
    vision: torch.Tensor,
    actor: torch.Tensor,
    target: torch.Tensor,
    *,
    microbatch_environments: int,
    full_environment_batch_per_timestep: bool = False,
    activation_checkpointing: bool = True,
    zero_auxiliary_object_loss: bool = False,
) -> float:
    """Backpropagate one Raven whole-window MSE.

    A Raven rollout tensor is environment-major: ``[B, T, ...]``.  The policy
    internally flattens ``B*T`` images before its CNN. Raven-fidelity mode must
    therefore call the policy once with the complete tensor: SyncBatchNorm
    reductions, normalized activations, running statistics, and their
    gradients all depend on the full ``B*T`` population. Time-sliced forwards
    are not equivalent even when their scalar losses are accumulated before a
    single optimizer step.

    Production installs :class:`IndexSafeMaxPool2d` before this call. That
    wrapper only shards the sample-independent pooling kernel that otherwise
    exceeds CUDA's signed 32-bit indexing limit; it never shards the policy or
    a normalization reduction. Legacy mode remains available solely for old
    run replay and splits only the environment dimension.
    """
    if vision.ndim != 5 or actor.ndim != 3 or target.ndim != 3:
        raise ValueError("learner tensors must be environment-major [B,T,...]")
    if not (vision.shape[:2] == actor.shape[:2] == target.shape[:2]):
        raise ValueError("vision, actor, and target B/T dimensions must match")
    if microbatch_environments < 1:
        raise ValueError("microbatch_environments must be positive")
    if type(activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if not activation_checkpointing and not full_environment_batch_per_timestep:
        raise ValueError("disabling recomputation requires the full B*T batch")
    if zero_auxiliary_object_loss and not full_environment_batch_per_timestep:
        raise ValueError('auxiliary-state continuation requires one full B*T forward')

    total_elements = target.numel()
    if total_elements < 1:
        raise ValueError("empty learner target")
    squared_error_sum = torch.zeros((), device=target.device, dtype=torch.float64)
    if zero_auxiliary_object_loss:
        if microbatch_environments != target.shape[0]:
            raise ValueError('diagnostic auxiliary continuation requires the full environment batch')
        def forward(images, observations):
            result = policy.action_mean_with_obj_pred({'vision_obs': images, 'actor_obs': observations})
            return result['action_mean'], result['predicted_obj_pos']
        prediction, objects = (checkpoint(forward, vision, actor, use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), _batch_norm_recompute_context(policy)))
            if activation_checkpointing else forward(vision, actor))
        if prediction.shape != target.shape or objects.shape != (*target.shape[:2], 3):
            raise ValueError('diagnostic action/object prediction shape differs')
        mse = F.mse_loss(prediction, target)
        # Object supervision has coefficient exactly zero. No object labels are
        # invented: the finite zero-weight branch only retains Raven's explicit
        # zero gradients and AdamW slots, rather than freezing/dropping the head.
        zero_auxiliary = objects.square().mean()
        if not bool(torch.isfinite(mse)) or not bool(torch.isfinite(zero_auxiliary)):
            raise RuntimeError('non-finite diagnostic action/object output')
        (mse + 0.0 * zero_auxiliary).backward()
        for parameter in policy.obj_pred_mlp.parameters():
            if parameter.grad is None or bool(torch.count_nonzero(parameter.grad)):
                raise RuntimeError('diagnostic auxiliary head requires explicit zero gradients')
        return float(mse.detach())
    if full_environment_batch_per_timestep:
        if microbatch_environments != target.shape[0]:
            raise ValueError(
                "full-environment timestep batching requires the complete "
                "environment batch"
            )
        prediction = (
            checkpointed_full_window_actions(policy, vision, actor)
            if activation_checkpointing
            else policy({"vision_obs": vision, "actor_obs": actor})["actions"]
        )
        full_sum = F.mse_loss(prediction, target, reduction="sum")
        if not bool(torch.isfinite(full_sum)):
            raise RuntimeError("non-finite DAgger training loss")
        (full_sum / total_elements).backward()
        squared_error_sum += full_sum.detach().to(torch.float64)
        del prediction, full_sum
        return float((squared_error_sum / total_elements).item())

    for start in range(0, target.shape[0], microbatch_environments):
        stop = min(start + microbatch_environments, target.shape[0])
        prediction = policy({
            "vision_obs": vision[start:stop],
            "actor_obs": actor[start:stop],
        })["actions"]
        micro_sum = torch.nn.functional.mse_loss(
            prediction, target[start:stop], reduction="sum"
        )
        if not bool(torch.isfinite(micro_sum)):
            raise RuntimeError("non-finite DAgger training loss")
        (micro_sum / total_elements).backward()
        squared_error_sum += micro_sum.detach().to(torch.float64)
        del prediction, micro_sum
    return float((squared_error_sum / total_elements).item())


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open('w') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def save_iteration_checkpoint(path: Path, base: dict, policy, optimizer, scheduler, *,
                              iteration_id: int, global_step: int, train_mse: float,
                              request: dict, rng_context: dict | None = None,
                              order_receipt: dict | None = None):
    learner_order.validate_parent(base, request)
    if learner_order.enabled(request):
        learner_order.validate_receipt(order_receipt, request)
    elif order_receipt is not None:
        raise ValueError('unexpected learner ordering receipt')
    output = dict(base)
    output.update({
        "schema": "roborender.dagger_student_checkpoint.v3",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "dagger_iteration_id": iteration_id,
        "global_step": global_step,
        "parent_checkpoint_sha256": request["input_checkpoint_sha256"],
        "dagger_run_id": request["run_id"],
        "dagger_round_id": request["round_id"],
        "dagger_job_id": request["job_id"],
        "dagger_train_chunk_ids": list(request["train_chunk_ids"]),
        "dagger_replay_manifest": request["parameters"].get("replay_manifest"),
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "train_mse": train_mse,
        **pending_candidate_metrics(request["parameters"].get(
            "metric_protocol", LEGACY_METRIC_PROTOCOL)),
    })
    admission = request['parameters'].get(diagnostic_parent.PARAMETER)
    chain = request['parameters'].get(diagnostic_reentry.PARAMETER)
    if admission is not None:
        diagnostic_parent.validate_admission(admission, request)
        if global_step != base['global_step']+1 or iteration_id != request['parameters']['iteration_id']:
            raise ValueError('diagnostic checkpoint must record exactly one accepted optimizer update')
        output.update(schema=diagnostic_parent.CHECKPOINT_SCHEMA,
            diagnostic_parent_admission=admission, production_checkpoint=False,
            production_promoted=False, selection='diagnostic_candidate_only_no_production_selection',
            diagnostic_origin_checkpoint_sha256=admission['scope']['parent_checkpoint_sha256'])
    elif uses_replay_optimizer(base):
        validate_replay_request(base,request)
        if global_step != base['global_step']+1 or iteration_id != request['parameters']['iteration_id']:
            raise ValueError('Raven replay checkpoint must record exactly one accepted optimizer update')
        lineage = base.get('replay_optimizer_lineage')
        if base['schema'] == 'roborender.bc_student_checkpoint.v4':
            lineage = dict(schema=REPLAY_LINEAGE_SCHEMA,
                bc_checkpoint_sha256=request['input_checkpoint_sha256'], bc_global_step=base['global_step'],
                training_batch_contract=base['training_batch_contract'],
                training_source_identity=base['training_source_identity'],
                optimizer_contract=base['optimizer_contract'])
            if base.get('training_sampling_contract', {}).get('schema') == 'roborender.bc-sampling.v2':
                lineage.update(training_sampling_contract=base['training_sampling_contract'],
                               training_cursor=base['training_cursor'])
        output.update(schema=REPLAY_DAGGER_SCHEMA, replay_optimizer_lineage=lineage,
                      selection='unselected_dagger_candidate')
    if chain is not None:
        diagnostic_reentry.validate_parent_metadata(base, request)
        diagnostic_reentry.validate_runtime_files(chain, worker_file=Path(__file__).resolve())
        output.update(diagnostic_reentry.checkpoint_metadata(request))
    rng = learner_rng.publish_rng(base, request, rng_context, global_step, file_sha256(Path(__file__)))
    if rng is not None:
        output['rng_state'], output['learner_rng_receipt'] = rng
    if order_receipt is not None:
        output['learner_order_receipt'] = order_receipt
    atomic_torch_save(output, path)
    return None if rng is None else rng[1]


def save_inference_sidecar(
    state_root: Path, full_checkpoint_sha256: str, policy
) -> Path:
    """Publish the policy-only bytes consumed by the eight rollout gateways.

    The full learner checkpoint also carries Adam state and training metadata.
    Deserializing all of it eight times merely to obtain ``policy_state_dict``
    made every checkpoint barrier unnecessarily expensive.  This sidecar is
    keyed by the already verified full-checkpoint digest and is published
    before the learner receipt can activate that digest.
    """

    path = state_root.resolve() / "inference_checkpoints" / (
        f"{full_checkpoint_sha256}.pt"
    )
    if not path.exists():
        atomic_torch_save({
            "schema": "roborender.student_inference_sidecar.v1",
            "full_checkpoint_sha256": full_checkpoint_sha256,
            "policy_state_dict": policy.state_dict(),
        }, path)
    return path


def validate_mse_parent_checkpoint(checkpoint: dict, *, requested_learning_rate: float | None = None,
                                   diagnostic_request: dict | None = None) -> None:
    """Admit exact continuation using loaded metadata, not another data scan.

    Historical direct BC116/DAgger keeps its70-slot vision/MLP optimizer and
    frozen object head. Versioned Raven-replay BC B16/B32 retains all78 slots,
    including eight zero-gradient auxiliary slots; std stays frozen. Replay
    metadata and actual model parameter order must agree without migration.
    Strict model load verifies the full model, and replay moments must be
    finite with exactly zero auxiliary moments.
    """
    schema = checkpoint.get("schema")
    diagnostic = diagnostic_request is not None
    chain = (diagnostic_request or {}).get('parameters', {}).get(diagnostic_reentry.PARAMETER)
    replay = uses_replay_optimizer(checkpoint)
    auxiliary = diagnostic or replay
    if chain is not None:
        diagnostic_reentry.validate_parent_metadata(checkpoint, diagnostic_request)
        diagnostic_reentry.validate_runtime_files(chain, worker_file=Path(__file__).resolve())
        if schema == diagnostic_parent.BC_SCHEMA:
            validate_bc32_diagnostic_metadata(checkpoint)
    elif diagnostic:
        admission = diagnostic_parent.validate_admission(
            diagnostic_request['parameters'].get(diagnostic_parent.PARAMETER), diagnostic_request)
        iteration = diagnostic_request['parameters']['iteration_id']
        expected_schema = diagnostic_parent.BC_SCHEMA if iteration == 1 else diagnostic_parent.CHECKPOINT_SCHEMA
        if schema != expected_schema or checkpoint.get('global_step') != iteration + 1:
            raise ValueError('diagnostic parent must be the exact BC32 two-update parent or preceding candidate')
        if iteration == 1:
            validate_bc32_diagnostic_metadata(checkpoint)
        elif (checkpoint.get('diagnostic_parent_admission') != admission
                or checkpoint.get('diagnostic_origin_checkpoint_sha256') != admission['scope']['parent_checkpoint_sha256']
                or checkpoint.get('dagger_iteration_id') != iteration - 1
                or checkpoint.get('dagger_run_id') != diagnostic_request['run_id']
                or checkpoint.get('dagger_round_id') != 1
                or checkpoint.get('production_checkpoint') is not False
                or checkpoint.get('production_promoted') is not False
                or checkpoint.get('selection') != 'diagnostic_candidate_only_no_production_selection'):
            raise ValueError('diagnostic candidate lineage/scope differs')
    if schema not in {"roborender.bc_student_checkpoint.v3", "roborender.bc_student_checkpoint.v4",
                      "roborender.dagger_student_checkpoint.v3", REPLAY_DAGGER_SCHEMA} and not diagnostic:
        raise ValueError("MSE continuation requires a supported complete training checkpoint")
    if schema == 'roborender.bc_student_checkpoint.v4':
        from pipeline.bc_training.stopping_gate import fixed_final_contract
        protocol = checkpoint.get('training_protocol')
        if not isinstance(protocol, dict):
            raise ValueError('BC-v4 requires its fixed-final training protocol')
        expected = fixed_final_contract(protocol.get('max_epochs'), budget_explicit=True)
        batch = checkpoint.get('training_batch_contract')
        epoch = checkpoint.get('epoch')
        if (canonical_json(protocol) != canonical_json(expected) or checkpoint.get('validation_enabled') is not False
                or checkpoint.get('selection') != 'last_completed_epoch'
                or type(epoch) is not int or epoch != expected['max_epochs']
                or not isinstance(batch, dict)
                or type(batch.get('updates_per_epoch')) is not int or batch['updates_per_epoch'] < 1
                or any(type(batch.get(key)) is not int for key in
                       ('chunks_per_update', 'frames_per_chunk', 'policy_forward_population'))
                or batch.get('chunks_per_update') not in ((16,32) if replay else (16,))
                or batch.get('frames_per_chunk') != 81
                or batch.get('policy_forward_population') != batch['chunks_per_update']*81
                or checkpoint.get('global_step') != epoch * batch['updates_per_epoch']):
            raise ValueError('BC-v4 parent must be the final complete fixed-budget admitted full Bx81 epoch')
        rng = checkpoint.get('rng_state')
        if not isinstance(rng, dict) or not {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'train_generator'} <= set(rng):
            raise ValueError('BC-v4 requires the saved complete RNG state')
    step = checkpoint.get("global_step")
    if type(step) is not int or step < 1:
        raise ValueError("MSE continuation requires a positive trained global_step")
    policy_state = checkpoint.get("policy_state_dict")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if (not isinstance(policy_state, dict) or any(not isinstance(name, str) for name in policy_state)
            or not isinstance(optimizer_state, dict)):
        raise ValueError("MSE continuation requires policy and AdamW state")
    prefixes = ("vision_module.", "mlp_module.", "obj_pred_mlp.") if auxiliary else ("vision_module.", "mlp_module.")
    parameters = [(name, value) for name, value in policy_state.items()
                  if name.startswith(prefixes)
                  and name.endswith((".weight", ".bias"))]
    if replay:
        names = validate_replay_optimizer_metadata(checkpoint)
        parameters = [(name, policy_state[name]) for name in names]
    groups, states = optimizer_state.get("param_groups"), optimizer_state.get("state")
    if not parameters or not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("MSE continuation requires the complete single-group Raven AdamW state")
    group = groups[0]
    ids = group.get("params") if isinstance(group, dict) else None
    if (not isinstance(ids, list) or any(type(value) is not int for value in ids)
            or ids != list(range(len(parameters))) or any(type(ident) is not int for ident in states)
            or set(states) != set(ids)):
        raise ValueError("AdamW parameter/state identities differ from the trained Raven policy")
    if (group.get("weight_decay") != 0.0 or group.get("amsgrad") is not False
            or not isinstance(group.get("betas"), (tuple, list))
            or tuple(group["betas"]) != (0.9, 0.999) or group.get("eps") != 1e-8
            or group.get("maximize") is not False):
        raise ValueError("parent optimizer is not the configured Raven AdamW")
    saved_lr = group.get("lr")
    if type(saved_lr) not in (int, float) or not math.isfinite(saved_lr) or saved_lr <= 0:
        raise ValueError("parent AdamW learning rate must be finite and positive")
    if requested_learning_rate is not None and requested_learning_rate != saved_lr:
        raise ValueError(
            "MSE exact continuation requires requested learning_rate to equal saved AdamW lr; "
            "intentional learning-rate changes require a separate supported protocol"
        )
    for ident, (name, parameter) in zip(ids, parameters):
        state = states[ident]
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"incomplete AdamW state for {name}")
        counter = state["step"]
        if isinstance(counter, torch.Tensor):
            if counter.device.type != "cpu" or counter.numel() != 1:
                raise ValueError("AdamW step must be a CPU scalar in the loaded checkpoint")
            counter = counter.item()
        if type(counter) not in (int, float) or not math.isfinite(counter) or counter != step:
            raise ValueError("AdamW step differs from checkpoint global_step")
        if not isinstance(parameter, torch.Tensor) or not parameter.is_floating_point():
            raise ValueError(f"invalid policy parameter metadata for {name}")
        for field in ("exp_avg", "exp_avg_sq"):
            moment = state[field]
            if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype or moment.device.type != "cpu"):
                raise ValueError(f"AdamW {field} metadata differs from policy parameter {name}")
            if auxiliary and (not bool(torch.isfinite(moment).all())
                    or (name.startswith('obj_pred_mlp.') and bool(torch.count_nonzero(moment)))):
                raise ValueError('diagnostic AdamW moments must be finite with exact zero auxiliary moments')
        if auxiliary and not bool(torch.isfinite(parameter).all()):
            raise ValueError('diagnostic policy parameters must be finite')
    if auxiliary and not any(name.startswith('obj_pred_mlp.') for name, _ in parameters):
        raise ValueError('diagnostic AdamW must retain the auxiliary object-head parameters')
    if schema in {"roborender.bc_student_checkpoint.v3", "roborender.bc_student_checkpoint.v4"} and checkpoint.get("lr_scheduler_state_dict") is not None:
        raise ValueError("BC-v3/v4 checkpoints do not carry a DAgger scheduler state")
    if schema == diagnostic_parent.BC_SCHEMA and checkpoint.get('lr_scheduler_state_dict') is not None:
        raise ValueError('BC32 diagnostic cannot invent a DAgger scheduler')
    if schema in ("roborender.dagger_student_checkpoint.v3", diagnostic_parent.CHECKPOINT_SCHEMA,
                  diagnostic_reentry.CHECKPOINT_SCHEMA, REPLAY_DAGGER_SCHEMA):
        scheduler = checkpoint.get("lr_scheduler_state_dict")
        if not isinstance(scheduler, dict):
            raise ValueError("DAgger continuation requires its saved scheduler state")
        epoch, calls = scheduler.get("last_epoch"), scheduler.get("_step_count")
        if (type(epoch) is not int or not 1 <= epoch <= step
                or type(calls) is not int or calls != epoch + 1):
            raise ValueError("DAgger scheduler step metadata is inconsistent")
        for field in ("base_lrs", "_last_lr"):
            values = scheduler.get(field)
            if (not isinstance(values, list) or len(values) != 1
                    or type(values[0]) not in (int, float) or not math.isfinite(values[0])
                    or values[0] != saved_lr):
                raise ValueError("DAgger constant scheduler learning rate differs from saved AdamW")
        if "initial_lr" in group and group["initial_lr"] != saved_lr:
            raise ValueError("DAgger constant scheduler initial_lr differs from saved AdamW")
        if scheduler.get("lr_lambdas") != [None]:
            raise ValueError("DAgger continuation requires the constant LambdaLR scheduler")


def validate_bc32_diagnostic_metadata(value):
    from pipeline.bc_training.qualify_raven_bc32_lifecycle import BOOTSTRAP_SHA, SPEC_SHA
    from pipeline.bc_training.train_bc_h200 import student_lifecycle_batch_fields, REPLAY_STUDENT_LIFECYCLE
    batch = value.get('training_batch_contract')
    required_batch = dict(student_lifecycle_batch_fields(REPLAY_STUDENT_LIFECYCLE, 32),
        chunks_per_update=32, frames_per_chunk=81, policy_forward_population=2592,
        microbatching=False, gradient_accumulation=False, training_chunk_count=64, updates_per_epoch=None)
    initial, sources = value.get('initial_student_proof', {}), value.get('source_identity')
    if (value.get('epoch') != 0 or value.get('global_step') != 2
            or value.get('qualification_optimizer_steps') != 2
            or value.get('training_protocol') != dict(schema=diagnostic_parent.BC_SCHEMA, optimizer_updates=2)
            or batch != required_batch or value.get('selection') != 'diagnostic_only_not_completed_epoch'
            or any(value.get(key) is not False for key in
                ('validation_enabled', 'completed_epoch', 'production_checkpoint', 'production_promoted'))
            or value.get('bootstrap_sha256') != BOOTSTRAP_SHA or value.get('policy_spec_sha256') != SPEC_SHA
            or initial.get('initial_state_exact_saved_checkpoint') is not True
            or initial.get('initial_checkpoint_sha256') != BOOTSTRAP_SHA
            or not isinstance(sources, dict) or not sources
            or not isinstance(value.get('rng_state'), dict)
            or not {'python','numpy','torch_cpu','torch_cuda','train_generator'} <= set(value['rng_state'])):
        raise ValueError('BC32 diagnostic checkpoint is not the actual bounded initial-student qualification')
    from pipeline.dagger.services.contracts import sha256_digest
    for name, sha in sources.items():
        if not isinstance(name, str) or not name.startswith('pipeline/'):
            raise ValueError('BC32 diagnostic source identity is invalid')
        sha256_digest(sha, 'diagnostic source sha256')
    sha256_digest(initial.get('initial_state_sha256'), 'initial state sha256')


def configure_diagnostic_optimizer_policy(policy, checkpoint):
    from pipeline.bc_training.train_bc_h200 import configure_raven_offline_student
    configure_raven_offline_student(policy)
    expected = [name for name in checkpoint['policy_state_dict']
        if name.startswith(('vision_module.', 'mlp_module.', 'obj_pred_mlp.')) and name.endswith(('.weight', '.bias'))]
    actual = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if actual != expected:
        raise ValueError('diagnostic model trainable order differs from saved AdamW slots')


def validate_replay_optimizer_metadata(checkpoint):
    """Bind the changed optimizer population explicitly; never infer/migrate slots."""
    from pipeline.bc_training.train_bc_h200 import student_lifecycle_batch_fields, REPLAY_STUDENT_LIFECYCLE
    batch = checkpoint.get('training_batch_contract')
    optimizer = checkpoint.get('optimizer_contract')
    sources = checkpoint.get('training_source_identity')
    if not isinstance(batch,dict) or not isinstance(optimizer,dict) or not isinstance(sources,dict) or not sources:
        raise ValueError('Raven replay requires complete batch/optimizer/source metadata')
    count=batch.get('chunks_per_update')
    if type(count) is not int or count not in (16,32):
        raise ValueError('Raven replay requires an admitted B16 or B32 parent')
    fields=student_lifecycle_batch_fields(REPLAY_STUDENT_LIFECYCLE,count)
    from pipeline.bc_training import cyclic_sampling
    sampling = checkpoint.get('training_sampling_contract', {})
    if not isinstance(sampling, dict) or sampling.get('schema') not in (None, 'roborender.bc-sampling.v1', cyclic_sampling.SCHEMA):
        raise ValueError('unsupported BC sampling provenance')
    cyclic = sampling.get('schema') == cyclic_sampling.SCHEMA
    lineage = checkpoint.get('replay_optimizer_lineage', {})
    if cyclic:
        bc_step = (lineage.get('bc_global_step') if checkpoint['schema'] == REPLAY_DAGGER_SCHEMA
                   and isinstance(lineage, dict) else checkpoint.get('global_step'))
        cyclic_sampling.validate_checkpoint_sampling(sampling, batch, checkpoint.get('training_cursor'), bc_step)
    if (any(batch.get(key)!=value for key,value in fields.items())
            or batch.get('frames_per_chunk')!=81 or batch.get('policy_forward_population')!=count*81
            or batch.get('activation_checkpointing') is not True or batch.get('drop_last') is not (not cyclic)
            or type(batch.get('training_chunk_count')) is not int
            or type(batch.get('updates_per_epoch')) is not int or batch['updates_per_epoch']<1
            or (not cyclic and batch['training_chunk_count']!=count*batch['updates_per_epoch'])):
        raise ValueError('Raven replay batch/lifecycle metadata differs')
    from pipeline.dagger.services.contracts import sha256_digest
    for name,sha in sources.items():
        if not isinstance(name,str) or not name.startswith('pipeline/'):
            raise ValueError('Raven replay source identity differs')
        sha256_digest(sha,'Raven replay source sha256')
    expected=dict(algorithm='AdamW',implementation='torch.optim.AdamW',learning_rate=.0002,
        weight_decay=0.,max_grad_norm=1.,betas=[.9,.999],eps=1e-8,amsgrad=False)
    if any(optimizer.get(key)!=value for key,value in expected.items()):
        raise ValueError('Raven replay optimizer contract differs')
    state=checkpoint['policy_state_dict']
    names=optimizer.get('trainable_parameters')
    exact_names=[name for name in state if name.endswith(('.weight','.bias'))]
    if (not isinstance(names,list) or names!=exact_names or len(set(names))!=len(names)
            or not any(name.startswith('obj_pred_mlp.') for name in names)):
        raise ValueError('Raven replay requires the exact declared trainable parameter order')
    if checkpoint['schema']==REPLAY_DAGGER_SCHEMA:
        lineage=checkpoint.get('replay_optimizer_lineage')
        lineage_keys={'schema','bc_checkpoint_sha256','bc_global_step',
                      'training_batch_contract','training_source_identity','optimizer_contract'}
        if cyclic:
            lineage_keys.update(('training_sampling_contract', 'training_cursor'))
        if (not isinstance(lineage,dict) or set(lineage)!=lineage_keys
                or lineage.get('schema')!=REPLAY_LINEAGE_SCHEMA
                or lineage.get('training_batch_contract')!=batch
                or lineage.get('training_source_identity')!=sources
                or lineage.get('optimizer_contract')!=optimizer
                or (cyclic and (lineage.get('training_sampling_contract') != sampling
                    or lineage.get('training_cursor') != checkpoint.get('training_cursor')))
                or type(lineage.get('bc_global_step')) is not int
                or not 1<=lineage['bc_global_step']<checkpoint['global_step']
                or type(checkpoint.get('dagger_iteration_id')) is not int
                or not 1<=checkpoint['dagger_iteration_id']<=100
                or type(checkpoint.get('dagger_round_id')) is not int
                or checkpoint['dagger_round_id'] not in (1,2,3,4)
                or checkpoint.get('selection')!='unselected_dagger_candidate'
                or checkpoint.get('global_step')!=lineage['bc_global_step']+
                    (checkpoint['dagger_round_id']-1)*100+checkpoint['dagger_iteration_id']):
            raise ValueError('Raven replay successor lost its immutable BC optimizer lineage')
        sha256_digest(lineage['bc_checkpoint_sha256'],'original BC replay parent')
    return names


def configure_replay_optimizer_policy(policy, checkpoint):
    from pipeline.bc_training.train_bc_h200 import configure_raven_offline_student
    names=validate_replay_optimizer_metadata(checkpoint)
    configure_raven_offline_student(policy)
    actual=[name for name,parameter in policy.named_parameters() if parameter.requires_grad]
    if actual!=names:
        raise ValueError('Raven replay model trainable order differs from saved AdamW slots')


def validate_replay_request(checkpoint, request):
    """New optimizer population runs only under the admitted full32 fixed100 protocol."""
    parameters=request['parameters']; iteration=parameters.get('iteration_id'); round_id=request.get('round_id')
    if (parameters.get('metric_protocol')!='fixed100_final_success'
            or parameters.get('raven_exact_full_batch') is not True
            or parameters.get('microbatch_environments')!=32
            or parameters.get('replay_manifest') is not None
            or len(request.get('train_chunk_ids',[]))!=32
            or type(iteration) is not int or not 1<=iteration<=100
            or type(round_id) is not int or round_id not in (1,2,3,4)):
        raise ValueError('Raven replay continuation requires exact full32x81 fixed100 request')
    if checkpoint['schema']=='roborender.bc_student_checkpoint.v4':
        if (round_id,iteration)!=(1,1):
            raise ValueError('new BC replay checkpoint may initialize only round1 update1')
    else:
        previous=(round_id,iteration-1) if iteration>1 else (round_id-1,100)
        if ((checkpoint.get('dagger_round_id'),checkpoint.get('dagger_iteration_id'))!=previous
                or checkpoint.get('dagger_run_id')!=request.get('run_id')):
            raise ValueError('Raven replay request must continue the exact preceding round/update')


def main() -> None:
    args = parse_args()
    request = json.loads(args.job.read_text())
    phase_telemetry = PhaseTelemetry(
        args.result.parent / "phase-timing.jsonl",
        latest_path=args.result.parent / "progress.json",
    )
    atexit.register(phase_telemetry.close)
    phase_identity = {
        "run_id": request["run_id"],
        "round_id": request["round_id"],
    }
    parameters = request["parameters"]
    allowed = {
        "learning_rate", "max_grad_norm", "num_mini_batches", "seed",
        "iteration_id", "wandb_mode", "wandb_project", "wandb_group", "wandb_run_name",
        "wandb_run_id", "wandb_resume", "wandb_extra_config",
        "deep_validation_samples", "microbatch_environments",
        "raven_exact_full_batch", "activation_checkpointing",
        "final_validation_iteration",
        "validation_interval_iterations",
        "replay_manifest",
        "metric_protocol",
        diagnostic_parent.PARAMETER,
        learner_rng.PARAMETER,
        learner_order.PARAMETER,
        diagnostic_reentry.PARAMETER,
    }
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unknown learner parameters: {sorted(unknown)}")
    learner_rng.enabled(request)  # Reject misspelled protocol before GPU work.
    learner_order.enabled(request)
    chain = parameters.get(diagnostic_reentry.PARAMETER)
    if chain is not None:
        diagnostic_reentry.validate_admission(chain, request)
        diagnostic_reentry.validate_runtime_files(chain, worker_file=Path(__file__).resolve())
    metric_protocol = parameters.get("metric_protocol", LEGACY_METRIC_PROTOCOL)
    metric_labels = metric_config(metric_protocol)
    learning_rate = float(parameters.get("learning_rate", 2e-4))
    max_grad_norm = float(parameters.get("max_grad_norm", 1.0))
    num_mini_batches = int(parameters.get("num_mini_batches", 1))
    seed = int(parameters.get("seed", 42))
    iteration_id = int(parameters.get("iteration_id", 0))
    final_validation_iteration = int(
        parameters.get("final_validation_iteration", iteration_id)
    )
    validation_interval_iterations = int(
        parameters.get("validation_interval_iterations", final_validation_iteration)
    )
    # Online prepare/commit already enforces the full tensor contract and
    # receipts.  Production should not rescan large immutable chunks before
    # each one-step update.
    deep_validation_samples = int(parameters.get("deep_validation_samples", 0))
    # Legacy runs used B=7 gradient accumulation. Raven-fidelity runs pass
    # B=28 and execute one activation-checkpointed B*T policy forward, so every
    # SyncBatchNorm layer sees the complete 28*81-frame population.
    microbatch_environments = int(parameters.get("microbatch_environments", 7))
    raven_exact_full_batch = parameters.get("raven_exact_full_batch", False)
    activation_checkpointing = parameters.get("activation_checkpointing", True)
    if type(activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if not activation_checkpointing and not raven_exact_full_batch:
        raise ValueError("disabling recomputation requires raven_exact_full_batch")
    if learning_rate <= 0 or max_grad_norm <= 0 or num_mini_batches != 1:
        raise ValueError("Raven DAgger requires positive LR/clip and num_mini_batches=1")
    if iteration_id < 1:
        raise ValueError("iteration_id must be positive")
    if final_validation_iteration < iteration_id:
        raise ValueError("final_validation_iteration cannot precede iteration_id")
    if validation_interval_iterations < 1:
        raise ValueError("validation_interval_iterations must be positive")
    if deep_validation_samples < 0:
        raise ValueError("deep_validation_samples must be non-negative")
    if microbatch_environments < 1:
        raise ValueError("microbatch_environments must be positive")
    if type(raven_exact_full_batch) is not bool:
        raise ValueError("raven_exact_full_batch must be boolean")
    wandb_extra_config = parameters.get("wandb_extra_config", {})
    if not isinstance(wandb_extra_config, dict):
        raise ValueError("wandb_extra_config must be an object")
    allowed_wandb_extra = {
        "smoke_test", "concurrent_environments", "rollout_steps_per_environment",
        "total_transitions", "rtx_gpu_index", "rtx_gpu_uuid", "rtx_actors",
        "rtx_collection_config_sha256", "raven_commit", "render_contract_sha256",
        "image3f_adapter_sha256", "image3f_engine_sha256",
        "student_action_probability", "teacher_action_probability",
        "renderer_worker_count",
        "raven_exact_full_batch", "raven_exact_raytraced_augmentation",
        "raven_random_initial_episode_lengths",
    }
    unknown_wandb_extra = set(wandb_extra_config) - allowed_wandb_extra
    if unknown_wandb_extra:
        raise ValueError(f"unknown wandb_extra_config fields: {sorted(unknown_wandb_extra)}")
    train_ids = request.get("train_chunk_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("train_chunk_ids must be non-empty")
    digest = request["input_checkpoint_sha256"]
    checkpoint_path = args.state_root.resolve() / "checkpoints" / f"{digest}.pt"
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != digest:
        raise RuntimeError("registered input checkpoint is missing or corrupt")
    with phase_telemetry.span(
        **phase_identity, phase="learner_checkpoint_and_model_initialization",
        details={"iteration_id": iteration_id},
    ) as initialization_timing:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        learner_order.validate_parent(checkpoint, request)
        diagnostic_admission = parameters.get(diagnostic_parent.PARAMETER)
        if diagnostic_admission is not None:
            diagnostic_parent.validate_admission(diagnostic_admission, request)
        if metric_protocol in (MSE_FINAL_TEST, 'fixed100_final_success') or uses_replay_optimizer(checkpoint):
            validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=learning_rate,
                diagnostic_request=request if diagnostic_admission is not None else None)
        if uses_replay_optimizer(checkpoint):
            validate_replay_request(checkpoint,request)
        spec = checkpoint.get("policy_spec")
        if not isinstance(spec, dict):
            raise RuntimeError("input checkpoint lacks embedded three-view policy_spec")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError("DAgger learner requires one H200 (SM90)")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
        memory_context = learner_memory.begin(torch.cuda) if chain is not None else None
        learner_memory.observe(memory_context, torch.cuda, 'before_model_construction')
        policy = build_policy(spec, checkpoint["policy_state_dict"], device)
        replay_optimizer = uses_replay_optimizer(checkpoint)
        if diagnostic_admission is not None:
            configure_diagnostic_optimizer_policy(policy, checkpoint)
        elif replay_optimizer:
            configure_replay_optimizer_policy(policy, checkpoint)
        index_safe_max_pools = 0
        if raven_exact_full_batch:
            index_safe_max_pools = install_index_safe_max_pool2d(policy)
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if isinstance(checkpoint.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if metric_protocol not in (MSE_FINAL_TEST, 'fixed100_final_success'):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
        if isinstance(checkpoint.get("lr_scheduler_state_dict"), dict):
            scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        initialization_timing["checkpoint_sha256"] = digest
        initialization_timing["index_safe_max_pools"] = index_safe_max_pools

    with phase_telemetry.span(
        **phase_identity, phase="learner_dataset_sample_check_and_load",
        total=(parameters.get("replay_manifest") or {}).get(
            "total_chunks", len(train_ids)
        ),
        details={
            "iteration_id": iteration_id,
            "dagger_train_chunks": len(train_ids),
            "replay_enabled": parameters.get("replay_manifest") is not None,
        },
    ) as load_timing:
        replay_manifest = parameters.get("replay_manifest")
        train_paths = replay_training_paths(
            args.state_root.resolve(), train_ids, replay_manifest,
            deep_validation_samples=deep_validation_samples, sample_seed=f"{seed}:train",
        )
        train_vision, train_actor, train_target = load_batch(train_paths, device, spec)
        if raven_exact_full_batch and microbatch_environments != train_vision.shape[0]:
            raise ValueError(
                "Raven-fidelity learner requires the full environment batch per timestep: "
                f"microbatch_environments={microbatch_environments}, "
                f"batch={train_vision.shape[0]}"
            )
        load_timing["train_frames"] = len(train_paths) * 81

    checkpoint_root = args.result.parent / "candidate_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    wandb_mode = parameters.get("wandb_mode", "disabled")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    wandb_resume = parameters.get("wandb_resume", "allow")
    if wandb_resume not in {"allow", "must", "never", "auto"}:
        raise ValueError("wandb_resume must be allow, must, never, or auto")
    if wandb_mode != "disabled":
        import wandb
        wandb_config = {
            "run_id": request["run_id"], "round_id": request["round_id"],
            "beta": CANONICAL_BETA[request["round_id"]],
            "input_checkpoint_sha256": request["input_checkpoint_sha256"],
            "train_chunks": len(train_paths),
            "frames_per_chunk": 81,
            "train_frames": len(train_paths) * 81,
            "train_chunk_ids": list(train_ids),
            "replay_manifest_sha256": (
                replay_manifest["sha256"] if replay_manifest is not None else None
            ),
            "replay_selected_counts": (
                replay_manifest["selected_counts"] if replay_manifest is not None
                else {"current_dagger": len(train_ids), "bc": 0, "prior_dagger": 0}
            ),
            "view_order": ["head_color", "hand_left", "hand_right"],
            "policy_action_dim": 11,
            "teacher_action_dim": 26,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "num_mini_batches": num_mini_batches,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
            "microbatch_environments": microbatch_environments,
            "raven_exact_full_batch": raven_exact_full_batch,
            "activation_checkpointing": activation_checkpointing,
            "batching_semantics": (
                "raven_exact_flattened_bxt"
                if raven_exact_full_batch
                else "legacy_environment_microbatch"
            ),
            "activation_memory": (
                "nonreentrant_full_policy_recompute"
                if raven_exact_full_batch and activation_checkpointing
                else "retained_full_policy_activations" if raven_exact_full_batch
                else "legacy_retained_microbatch"
            ),
            "index_safe_max_pools": index_safe_max_pools,
            "deep_validation_samples": deep_validation_samples,
            "seed": seed,
            "iteration_id": iteration_id,
            "final_validation_iteration": final_validation_iteration,
            "validation_interval_iterations": validation_interval_iterations,
            **metric_labels,
        }
        overlap = set(wandb_config) & set(wandb_extra_config)
        if overlap:
            raise ValueError(f"wandb_extra_config cannot replace canonical fields: {sorted(overlap)}")
        wandb_config.update(wandb_extra_config)
        wandb_run = wandb.init(
            project=parameters.get("wandb_project", "dagger"),
            group=parameters.get("wandb_group", f"dagger_{request['round_id']}"),
            name=parameters.get("wandb_run_name", request["job_id"]),
            id=parameters.get("wandb_run_id"),
            resume=wandb_resume if parameters.get("wandb_run_id") else None,
            mode=wandb_mode,
            config=wandb_config,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("replay/*", step_metric="global_step")

    global_step = int(checkpoint.get("global_step", 0))
    gradient_norm = float("nan")
    try:
        # Process/model/dataset/logging initialization may consume randomness.
        # Restore only now; capture post-update state before publishing below.
        # This is learner-only state, never a claim of simulator restoration.
        rng_context = learner_rng.prepare_restore(checkpoint, request, file_sha256(Path(__file__)))
        (train_vision, train_actor, train_target), order_receipt = learner_order.permute_batch(
            request, train_vision, train_actor, train_target)
        learner_memory.observe(memory_context, torch.cuda, 'before_update')
        train_started = time.perf_counter()
        with phase_telemetry.span(
            **phase_identity, phase="learner_iteration_training",
            details={"iteration_id": iteration_id, "global_step_before": global_step},
        ) as train_timing:
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            train_mse = backward_global_mse_in_environment_microbatches(
                policy,
                train_vision,
                train_actor,
                train_target,
                microbatch_environments=microbatch_environments,
                full_environment_batch_per_timestep=raven_exact_full_batch,
                activation_checkpointing=activation_checkpointing,
                zero_auxiliary_object_loss=diagnostic_admission is not None or replay_optimizer,
            )
            finite_gradients = True
            nonzero_gradients = False
            for parameter in trainable:
                if parameter.grad is not None:
                    finite_gradients &= bool(torch.isfinite(parameter.grad).all())
                    nonzero_gradients |= bool(torch.count_nonzero(parameter.grad))
            if not finite_gradients or not nonzero_gradients:
                raise RuntimeError("DAgger gradients are non-finite or all zero")
            learner_memory.observe(memory_context, torch.cuda, 'after_backward_before_optimizer')
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
            optimizer.step()
            scheduler.step()
            global_step += 1
            learner_memory.observe(memory_context, torch.cuda, 'after_update')
            if diagnostic_admission is not None or replay_optimizer:
                for parameter in trainable:
                    state = optimizer.state[parameter]
                    if (int(state['step']) != global_step
                            or any(not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq'))):
                        raise RuntimeError('diagnostic optimizer slots did not continue exactly')
                for parameter in policy.obj_pred_mlp.parameters():
                    if any(bool(torch.count_nonzero(optimizer.state[parameter][k])) for k in ('exp_avg','exp_avg_sq')):
                        raise RuntimeError('diagnostic auxiliary optimizer moments changed')
            train_timing.update({
                "iteration_id": iteration_id, "global_step": global_step,
                "train_mse": train_mse, "gradient_norm": gradient_norm,
                "microbatch_environments": microbatch_environments,
                "batching_semantics": (
                    "raven_exact_flattened_bxt"
                    if raven_exact_full_batch
                    else "legacy_environment_microbatch"
                ),
                "activation_memory": (
                    "nonreentrant_full_policy_recompute"
                    if raven_exact_full_batch and activation_checkpointing
                    else "retained_full_policy_activations" if raven_exact_full_batch
                    else "legacy_retained_microbatch"
                ),
                "index_safe_max_pools": index_safe_max_pools,
            })
        train_seconds = time.perf_counter() - train_started
        output_path = checkpoint_root / (
            f"iteration_{iteration_id:04d}_step_{global_step:06d}.pt"
        )
        checkpoint_started = time.perf_counter()
        with phase_telemetry.span(
            **phase_identity, phase="learner_candidate_checkpoint",
            details={"iteration_id": iteration_id, "global_step": global_step},
        ) as checkpoint_timing:
            rng_receipt = save_iteration_checkpoint(
                output_path, checkpoint, policy, optimizer, scheduler,
                iteration_id=iteration_id, global_step=global_step,
                train_mse=train_mse, request=request, rng_context=rng_context,
                order_receipt=order_receipt,
            )
            checkpoint_timing["checkpoint_path"] = str(output_path.resolve())
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        if wandb_run is not None:
            replay_counts = (
                replay_manifest["selected_counts"] if replay_manifest is not None
                else {"current_dagger": len(train_ids), "bc": 0, "prior_dagger": 0}
            )
            wandb_run.log({
                "iteration_id": iteration_id, "global_step": global_step,
                "train/iteration_mse": train_mse,
                "train/mse_step": train_mse,
                "train/mse": train_mse,
                "timing/train_iteration_seconds": train_seconds,
                "timing/checkpoint_seconds": checkpoint_seconds,
                "replay/current_dagger_chunks": replay_counts["current_dagger"],
                "replay/bc_chunks": replay_counts["bc"],
                "replay/prior_dagger_chunks": replay_counts["prior_dagger"],
            })
            wandb_run.summary.update({
                "last_iteration_id": iteration_id,
                "last_train_mse": train_mse,
                **candidate_summary_status(metric_protocol),
                "last_replay_manifest_sha256": (
                    replay_manifest["sha256"] if replay_manifest is not None else None
                ),
            })
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    output_sha256 = file_sha256(output_path)
    inference_sidecar = save_inference_sidecar(
        args.state_root, output_sha256, policy
    )
    receipt = {
        "schema": "roborender.dagger_learner_candidate_receipt.v3",
        "output_checkpoint_path": str(output_path.resolve()),
        "output_checkpoint_sha256": output_sha256,
        "inference_sidecar_path": str(inference_sidecar.resolve()),
        "parent_checkpoint_sha256": digest,
        "global_step": global_step,
        "parent_global_step": int(checkpoint.get("global_step", 0)),
        "optimizer_state_preserved": bool(checkpoint.get("optimizer_state_dict", {}).get("state")),
        "train_chunks": len(train_paths),
        "train_frames": len(train_paths) * 81,
        "replay_manifest_sha256": (
            replay_manifest["sha256"] if replay_manifest is not None else None
        ),
        "replay_selected_counts": (
            replay_manifest["selected_counts"] if replay_manifest is not None
            else {"current_dagger": len(train_ids), "bc": 0, "prior_dagger": 0}
        ),
        "iteration_id": iteration_id,
        "train_mse": train_mse,
        **pending_candidate_metrics(metric_protocol),
        "gradient_norm_before_clip": gradient_norm,
        "batching_semantics": (
            "raven_exact_flattened_bxt"
            if raven_exact_full_batch
            else "legacy_environment_microbatch"
        ),
        "activation_memory": (
            "nonreentrant_full_policy_recompute"
            if raven_exact_full_batch and activation_checkpointing
            else "retained_full_policy_activations" if raven_exact_full_batch
            else "legacy_retained_microbatch"
        ),
        "activation_checkpointing": activation_checkpointing,
        "index_safe_max_pools": index_safe_max_pools,
        "device": torch.cuda.get_device_name(0),
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
    }
    if rng_receipt is not None:
        receipt['learner_rng_receipt'] = rng_receipt
    if order_receipt is not None:
        receipt['learner_order_receipt'] = order_receipt
        receipt['train_chunk_ids'] = list(train_ids)
    if memory_context is not None:
        receipt['learner_memory'] = memory_context
    if diagnostic_admission is not None:
        receipt.update(diagnostic_only=True, production_promoted=False,
            diagnostic_scope_sha256=diagnostic_admission['scope_sha256'],
            diagnostic_origin_checkpoint_sha256=diagnostic_admission['scope']['parent_checkpoint_sha256'],
            output_checkpoint_schema=diagnostic_parent.CHECKPOINT_SCHEMA,
            optimizer_parameter_states=len(optimizer.state), auxiliary_object_head_zero_moments=True)
    elif replay_optimizer:
        receipt.update(output_checkpoint_schema=REPLAY_DAGGER_SCHEMA,
            optimizer_parameter_states=len(optimizer.state), auxiliary_object_head_zero_moments=True,
            optimizer_continuation='raven_replay_auxiliary_slots_v1')
    chain = parameters.get(diagnostic_reentry.PARAMETER)
    if chain is not None:
        diagnostic_reentry.validate_runtime_files(chain, worker_file=Path(__file__).resolve())
        receipt.update(
            diagnostic_learner_chain_admission_sha256=diagnostic_parent.digest(chain),
            learner_execution=chain['execution'],
            output_checkpoint_schema=diagnostic_reentry.checkpoint_metadata(request)['schema'])
    atomic_json(receipt, args.result)
    phase_telemetry.emit(
        **phase_identity,
        phase="learner_job",
        event="end",
        status="completed",
        completed=1,
        total=1,
        details={
            "iteration_id": iteration_id,
            "train_mse": train_mse,
            "validation_status": "pending_closed_loop_evaluation",
        },
    )
    phase_telemetry.close()
    print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
