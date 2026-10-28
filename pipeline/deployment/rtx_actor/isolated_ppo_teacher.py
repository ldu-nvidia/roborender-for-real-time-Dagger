"""Private original-PPO teacher ownership for a fresh resident lifecycle.

This module performs no model loading, forward, physics or device initialization.
Build the private model before the collection teacher's first forward. Never
silently fall back to a shared teacher if copying or attestation fails.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import copy
import random
import types

_ABSENT = object()
_RENDER_BINDINGS = (
    "_success_observation_transform", "_rendered_success_evaluation_binding",
    "_rendered_paired_baseline", "_require_rendered_success_observations",
)


class _Snapshot:
    """Identity, storage, bytes and Python attributes, including non-state tensors.

    Traversing instance dictionaries also covers RMS flags, noise controls and
    rollout distributions/buffers. Cycles (including OmegaConf parents) are
    recorded, not followed twice. Unknown opaque mutable state fails closed.
    """
    def __init__(self, value):
        import numpy as np
        import torch

        self.mutable_ids, self.storages, self.tensors = set(), set(), {}
        seen = set()

        def visit(item, path):
            if item is None or type(item) in (bool, int, str, bytes):
                return (type(item), item)
            if type(item) is float:
                return (float, item.hex())
            if isinstance(item, (type, types.FunctionType, types.BuiltinFunctionType)):
                return ("callable", id(item))
            if isinstance(item, types.MethodType):
                return ("method", id(item.__func__), visit(item.__self__, path + ".__self__"))
            if isinstance(item, (torch.dtype, torch.device, torch.layout)):
                return (type(item), str(item))
            if id(item) in seen:
                return ("reference", id(item))
            seen.add(id(item))
            if isinstance(item, torch.Tensor):
                if item.layout != torch.strided or item.device.type == "meta":
                    raise RuntimeError("unsupported teacher tensor at " + path)
                self.mutable_ids.add(id(item))
                storage = (str(item.device), item.untyped_storage().data_ptr())
                if item.untyped_storage().nbytes():
                    self.storages.add(storage)
                self.tensors[path] = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone()
                return ("tensor", id(item), storage, item.data_ptr(), tuple(item.shape),
                        tuple(item.stride()), item.storage_offset(), str(item.dtype),
                        item.requires_grad, item._version,
                        visit(item.grad, path + ".grad") if item.is_leaf else "nonleaf")
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    raise RuntimeError("unsupported object array at " + path)
                self.mutable_ids.add(id(item))
                return ("array", id(item), str(item.dtype), item.shape, item.strides, item.tobytes())
            if isinstance(item, (tuple, frozenset)):
                values = list(item) if isinstance(item, tuple) else sorted(item, key=repr)
                return (type(item), tuple(visit(child, path + f"[{index}]")
                                         for index, child in enumerate(values)))
            self.mutable_ids.add(id(item))
            if isinstance(item, (list, set)):
                values = item if isinstance(item, list) else sorted(item, key=repr)
                return (type(item), id(item), tuple(visit(child, path + f"[{index}]")
                                                    for index, child in enumerate(values)))
            if isinstance(item, Mapping) and type(item).__module__ in ("builtins", "collections"):
                return (type(item), id(item), tuple((visit(key, path + ".key"),
                    visit(child, path + f"[{key!r}]")) for key, child in item.items()))
            if hasattr(item, "__dict__"):
                return (type(item), id(item), tuple((key, visit(child, path + "." + key))
                    for key, child in sorted(vars(item).items())))
            raise RuntimeError("unsupported opaque teacher attribute at " + path)

        self.signature = visit(value, "teacher")

    def assert_equal(self, value):
        import torch

        actual = _Snapshot(value)
        if actual.signature != self.signature or actual.tensors.keys() != self.tensors.keys():
            raise RuntimeError("shared teacher identity, storage, mode or attributes changed")
        for name, expected in self.tensors.items():
            if not torch.equal(expected, actual.tensors[name]):
                raise RuntimeError("shared teacher tensor bytes changed: " + name)


def _rng_state():
    import numpy as np
    import torch

    # is_initialized is read-only; do not initialize CUDA to inspect its RNG.
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _rng_equal(left, right):
    import numpy as np
    import torch

    return (left[0] == right[0] and left[1][0] == right[1][0]
            and np.array_equal(left[1][1], right[1][1]) and left[1][2:] == right[1][2:]
            and torch.equal(left[2], right[2])
            and ((left[3] is None and right[3] is None)
                 or (left[3] is not None and right[3] is not None
                     and len(left[3]) == len(right[3])
                     and all(torch.equal(a, b) for a, b in zip(left[3], right[3])))))


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _disjoint(shared, private):
    a, b = _Snapshot(shared), _Snapshot(private)
    if a.mutable_ids & b.mutable_ids or a.storages & b.storages:
        raise RuntimeError("private PPO teacher aliases shared teacher state or storage")


def _copy_with_private_config_parents(shared):
    """Copy OmegaConf's whole reachable graph before copying the module.

    OmegaConf deliberately retains a config node's original _parent during
    deepcopy. Raven's nested algo/backbone configs would therefore retain the
    collection configuration even though every parameter is private. Keep all
    nodes, ancestor links and interpolation topology, but rebind copied parents
    through the same deepcopy memo. Never detach or mutate the source parents.
    """
    from omegaconf.base import Node
    import torch

    nodes, seen = [], set()

    def collect(value):
        if id(value) in seen or isinstance(value, (torch.Tensor, type,
                types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            return
        seen.add(id(value))
        if isinstance(value, Node):
            nodes.append(value)
        if isinstance(value, Mapping) and type(value).__module__ in ("builtins", "collections"):
            for key, child in value.items():
                collect(key)
                collect(child)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                collect(child)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                collect(child)

    collect(shared)
    memo = {}
    for node in nodes:
        copy.deepcopy(node, memo)
    for node in nodes:
        parent = node._get_parent()
        memo[id(node)]._set_parent(None if parent is None else memo[id(parent)])
    return copy.deepcopy(shared, memo)


def build_isolated_teacher(trainer):
    """Return a frozen/eval private copy; caller caches it once at owner startup.

    This intentionally rejects post-forward/non-pristine teachers. A deepcopy
    failure is an admission failure, never permission to copy a partial state,
    reconstruct a model with fresh random parameters or reuse the shared model.
    """
    import torch

    shared = trainer.ref_model
    if not isinstance(shared, torch.nn.Module):
        raise RuntimeError("shared teacher must be an actual torch module")
    if (hasattr(shared, "_original_ppo_std_policy")
            or getattr(shared, "distribution", None) is not None
            or getattr(shared, "steps", 0) != 0
            or len(getattr(shared, "obs_dict_buffer", {})) != 0
            or getattr(shared, "dones_buffer", None) is not None):
        raise RuntimeError("private PPO teacher requires pristine pre-forward shared teacher")
    before, rng = _Snapshot(shared), _rng_state()
    try:
        try:
            private = _copy_with_private_config_parents(shared)
        except Exception as exc:
            raise RuntimeError("private PPO teacher deepcopy failed; no shared fallback") from exc
        _disjoint(shared, private)
        private.eval()
        private.requires_grad_(False)
        source, target = shared.state_dict(), private.state_dict()
        if source.keys() != target.keys() or any(
                not torch.equal(source[name], target[name]) for name in source):
            raise RuntimeError("private PPO teacher initial state differs")
        if any(module.training for module in private.modules()) or any(
                parameter.requires_grad for parameter in private.parameters()):
            raise RuntimeError("private PPO teacher must be frozen and eval")
        _disjoint(shared, private)
        return private
    finally:
        # Even a failed/malicious deepcopy may not quietly consume startup RNG.
        changed_rng = not _rng_equal(rng, _rng_state())
        if changed_rng:
            _restore_rng(rng)
        before.assert_equal(trainer.ref_model)
        if changed_rng:
            raise RuntimeError("private PPO teacher construction consumed RNG; admission rejected")


def _require_no_rendering(evaluator):
    if any(getattr(evaluator, name, None) not in (None, False) for name in _RENDER_BINDINGS):
        raise RuntimeError("teacher-only execution rejects renderer/observation bindings")


def _replace_attribute(owner, name, value):
    previous = vars(owner).get(name, _ABSENT)
    setattr(owner, name, value)

    def restore():
        if previous is _ABSENT:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)
    return restore


@contextmanager
def teacher_only_execution(evaluator, private_teacher, *, shared_trainer):
    """Bind only the evaluator and prove shared-teacher preservation on exit.

    Guards replace the actual student forward entrypoints (including direct
    .forward calls), not only nn.Module hooks. The context requires serialized
    resident ownership: it is not safe to use the student concurrently.
    A caught forbidden call still makes the final receipt fail closed.
    """
    import torch

    if evaluator is shared_trainer or private_teacher is shared_trainer.ref_model:
        raise RuntimeError("teacher-only execution requires a separate evaluator and teacher")
    if not isinstance(private_teacher, torch.nn.Module):
        raise RuntimeError("private PPO teacher must be an actual torch module")
    _require_no_rendering(evaluator)
    _disjoint(shared_trainer.ref_model, private_teacher)
    before = _Snapshot(shared_trainer.ref_model)
    receipt = {"schema": "roborender.isolated-ppo-teacher-execution.v1",
               "student_forward_calls": 0, "renderer_calls": 0,
               "distill_policy_step_calls": 0, "teacher_rollout_calls": 0,
               "shared_teacher_unchanged": False, "private_teacher_used": False}
    restore = []

    def forbid(counter):
        def blocked(*_args, **_kwargs):
            receipt[counter] += 1
            raise RuntimeError("teacher-only execution forbids " + counter)
        return blocked

    original_rollout = private_teacher.rollout

    def rollout(*args, **kwargs):
        if evaluator.ref_model is not private_teacher:
            raise RuntimeError("private PPO teacher binding changed")
        receipt["teacher_rollout_calls"] += 1
        receipt["private_teacher_used"] = True
        return original_rollout(*args, **kwargs)

    try:
        restore.append(_replace_attribute(evaluator, "ref_model", private_teacher))
        restore.append(_replace_attribute(private_teacher, "rollout", rollout))
        models = {id(model): model for model in
                  (getattr(evaluator, "policy_model", None), getattr(evaluator, "unwrapped_model", None))
                  if model is not None}
        for model in models.values():
            if model in (private_teacher, shared_trainer.ref_model):
                raise RuntimeError("student entrypoint aliases a teacher")
            restore.append(_replace_attribute(model, "forward", forbid("student_forward_calls")))
        restore.append(_replace_attribute(evaluator, "policy_step", forbid("distill_policy_step_calls")))
        for name in ("_image_dagger_relabel", "_videogen_relabel"):
            restore.append(_replace_attribute(evaluator, name, forbid("renderer_calls")))
        yield receipt
        _require_no_rendering(evaluator)
        if evaluator.ref_model is not private_teacher:
            raise RuntimeError("private PPO teacher binding changed")
        if any(receipt[name] for name in
               ("student_forward_calls", "renderer_calls", "distill_policy_step_calls")):
            raise RuntimeError("teacher-only forbidden call was caught by caller")
    finally:
        for undo in reversed(restore):
            undo()
        before.assert_equal(shared_trainer.ref_model)
        receipt["shared_teacher_unchanged"] = True


def validate_execution_receipt(receipt, *, vector_steps):
    """Admit only an actually completed, guarded teacher-only action sequence."""
    expected = {"schema": "roborender.isolated-ppo-teacher-execution.v1",
                "student_forward_calls": 0, "renderer_calls": 0,
                "distill_policy_step_calls": 0, "teacher_rollout_calls": vector_steps,
                "shared_teacher_unchanged": True, "private_teacher_used": True}
    if (type(vector_steps) is not int or vector_steps <= 0 or type(receipt) is not dict
            or set(receipt) != set(expected)
            or any(type(receipt[key]) is not type(value) or receipt[key] != value
                   for key, value in expected.items())):
        raise ValueError("invalid isolated PPO teacher execution receipt or vector-step count")
    return dict(receipt)
