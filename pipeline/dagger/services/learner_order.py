"""Raven's one environment permutation per full 81-step learner update.

This protocol is opt-in for fresh, source-bound campaigns. Historical requests
keep their recorded semantics; a new-protocol checkpoint cannot downgrade.
Time within each environment is never shuffled.
"""
from __future__ import annotations

import hashlib
import json

PARAMETER = 'learner_order_protocol'
PROTOCOL = 'roborender.raven-environment-randperm.v1'
CONFIG_FIELD = 'learner_protocol'


def enabled(request):
    parameters = request.get('parameters', {})
    value = parameters.get(PARAMETER)
    if value is None:
        return False
    if value != PROTOCOL:
        raise ValueError('unknown learner ordering protocol')
    if (parameters.get('learner_rng_protocol') != 'roborender.learner-rng-continuation.v1'
            or parameters.get('raven_exact_full_batch') is not True
            or parameters.get('num_mini_batches') != 1
            or parameters.get('replay_manifest') is not None):
        raise ValueError('Raven ordering requires saved RNG and one full online batch without replay')
    ids = request.get('train_chunk_ids')
    if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids)
            or len(set(ids)) != len(ids)
            or parameters.get('microbatch_environments') != len(ids)):
        raise ValueError('Raven ordering requires unique lane-ordered chunks and a complete batch')
    return True


def _ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def validate_receipt(receipt, request):
    ids = request['train_chunk_ids']
    if not isinstance(receipt, dict):
        raise ValueError('missing actual learner permutation receipt')
    indices = receipt.get('environment_indices')
    if (not isinstance(indices, list) or any(type(x) is not int for x in indices)
            or sorted(indices) != list(range(len(ids)))):
        raise ValueError('learner indices must be a permutation of all environments')
    expected = dict(schema=PROTOCOL, environment_indices=indices,
        input_chunk_ids_sha256=_ids_sha(ids),
        permuted_chunk_ids_sha256=_ids_sha([ids[i] for i in indices]),
        temporal_order='unchanged', rollout_steps=81)
    if receipt != expected:
        raise ValueError('learner permutation receipt differs from the admitted batch')
    return receipt


def validate_parent(base, request):
    from . import learner_rng

    previous = base.get('learner_order_receipt')
    if not enabled(request):
        if previous is not None:
            raise ValueError('cannot downgrade Raven learner ordering')
        return
    if previous is None:
        if base.get('schema') not in learner_rng.INITIAL_SCHEMAS:
            raise ValueError('new learner ordering requires a fresh BC parent, not a legacy DAgger resume')
    else:
        validate_receipt(previous, dict(train_chunk_ids=base.get('dagger_train_chunk_ids', [])))


def permute_batch(request, vision, actor, target):
    """Call after RNG restore, once, immediately before the learning forward."""
    if not enabled(request):
        return (vision, actor, target), None
    import torch

    count = len(request['train_chunk_ids'])
    if any(x.shape[:2] != (count, 81) or x.device != vision.device
           for x in (vision, actor, target)):
        raise ValueError('learner tensors must share device and full B x 81 lane/time dimensions')
    # Same operation and device as PPOTrainer.train's nonrecurrent b_inds.
    indices = torch.randperm(count, device=vision.device)
    order = indices.cpu().tolist()
    ids = request['train_chunk_ids']
    receipt = dict(schema=PROTOCOL, environment_indices=order,
        input_chunk_ids_sha256=_ids_sha(ids),
        permuted_chunk_ids_sha256=_ids_sha([ids[i] for i in order]),
        temporal_order='unchanged', rollout_steps=81)
    return tuple(x[indices] for x in (vision, actor, target)), receipt
