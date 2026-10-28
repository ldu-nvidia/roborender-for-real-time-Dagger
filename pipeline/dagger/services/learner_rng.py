"""Explicit learner-only RNG continuation; never a native simulator snapshot.

Legacy jobs retain their historical per-job seed behavior. New jobs opt in and
must restore after process/model/logging initialization, immediately before the
learning forward. Their checkpoint replaces, rather than inherits, BC RNG bytes.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

import numpy as np
import torch

PARAMETER = 'learner_rng_protocol'
PROTOCOL = 'roborender.learner-rng-continuation.v1'
FIELDS = {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'train_generator'}
INITIAL_SCHEMAS = {'roborender.bc_student_checkpoint.v4',
                   'roborender.bc32-lifecycle-qualification.v1'}


def enabled(request):
    value = request.get('parameters', {}).get(PARAMETER)
    if value is None:
        return False
    if value != PROTOCOL:
        raise ValueError('unknown learner RNG protocol')
    return True


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('exact SHA256 identity required for learner RNG continuation')
    return value


def _byte_state(value):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != torch.uint8 or value.ndim != 1
            or not 1 <= value.numel() <= 1_000_000):
        raise ValueError('bounded CPU byte tensor required for RNG state')
    return value.detach().contiguous().numpy().tobytes().hex()


def state_sha256(state):
    """Validate without changing any global RNG, then hash exact state values."""
    if not isinstance(state, dict) or set(state) != FIELDS:
        raise ValueError('complete Python/NumPy/Torch/device/train-generator RNG state required')
    random.Random(0).setstate(state['python'])
    numpy_state = state['numpy']
    if (not isinstance(numpy_state, tuple) or len(numpy_state) != 5
            or numpy_state[0] != 'MT19937' or not isinstance(numpy_state[1], np.ndarray)
            or numpy_state[1].dtype != np.uint32 or numpy_state[1].shape != (624,)):
        raise ValueError('exact NumPy MT19937 state required')
    np.random.RandomState(0).set_state(numpy_state)
    cpu = _byte_state(state['torch_cpu'])
    generator = _byte_state(state['train_generator'])
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    torch.Generator(device='cpu').set_state(state['train_generator'])
    cuda = state['torch_cuda']
    if not isinstance(cuda, list) or len(cuda) > 8:
        raise ValueError('bounded device RNG state list required')
    value = dict(python=state['python'],
        numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        torch_cpu=cpu, torch_cuda=[_byte_state(row) for row in cuda], train_generator=generator)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def capture_rng(train_generator):
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[row.clone() for row in torch.cuda.get_rng_state_all()],
        train_generator=train_generator.detach().clone())
    state_sha256(state)
    return state


def _check_parent(base, request):
    mode = enabled(request)
    receipt = base.get('learner_rng_receipt')
    if not mode:
        if receipt is not None:
            raise ValueError('cannot silently downgrade a checkpoint with learner RNG continuation')
        return None
    step = base.get('global_step')
    if type(step) is not int or step < 1:
        raise ValueError('positive trained parent global_step required')
    state = base.get('rng_state')
    digest = state_sha256(state)
    if receipt is None:
        if base.get('schema') not in INITIAL_SCHEMAS:
            raise ValueError('legacy DAgger RNG is not a saved learner state; explicit fresh BC parent required')
    elif (not isinstance(receipt, dict) or receipt.get('schema') != PROTOCOL
            or receipt.get('output_rng_sha256') != digest or receipt.get('global_step') != step
            or receipt.get('train_generator_usage') != 'preserved_unused_by_dagger'
            or receipt.get('parent_checkpoint_sha256') != base.get('parent_checkpoint_sha256')):
        raise ValueError('parent learner RNG receipt does not bind its actual state/checkpoint step')
    else:
        for key in ('input_rng_sha256', 'output_rng_sha256', 'parent_checkpoint_sha256', 'worker_file_sha256'):
            _digest(receipt.get(key))
    if len(state['torch_cuda']) != torch.cuda.device_count():
        raise ValueError('visible CUDA topology differs from saved learner RNG state')
    return digest


def prepare_restore(base, request, worker_sha):
    """Call after all incidental initialization, immediately before learning."""
    digest = _check_parent(base, request)
    if digest is None:
        return None
    worker_sha = _digest(worker_sha)
    state = base['rng_state']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    restored = capture_rng(state['train_generator'])
    if state_sha256(restored) != digest:
        raise RuntimeError('learner RNG restoration failed exact readback')
    return dict(schema=PROTOCOL, input_rng_sha256=digest, parent_global_step=base['global_step'],
        parent_checkpoint_sha256=_digest(request['input_checkpoint_sha256']), worker_file_sha256=worker_sha)


def publish_rng(base, request, context, global_step, worker_sha):
    """Capture the current post-update state before publishing the checkpoint."""
    if not enabled(request):
        if context is not None or base.get('learner_rng_receipt') is not None:
            raise ValueError('unexpected or downgraded learner RNG context')
        return None
    expected = dict(schema=PROTOCOL, input_rng_sha256=state_sha256(base['rng_state']),
        parent_global_step=base['global_step'], parent_checkpoint_sha256=_digest(request['input_checkpoint_sha256']),
        worker_file_sha256=_digest(worker_sha))
    if context != expected or type(global_step) is not int or global_step != base['global_step'] + 1:
        raise ValueError('learner RNG publication requires actual restored parent and one update')
    state = capture_rng(base['rng_state']['train_generator'])
    receipt = dict(schema=PROTOCOL, input_rng_sha256=context['input_rng_sha256'],
        output_rng_sha256=state_sha256(state), parent_checkpoint_sha256=context['parent_checkpoint_sha256'],
        global_step=global_step, worker_file_sha256=worker_sha,
        train_generator_usage='preserved_unused_by_dagger')
    return state, receipt
