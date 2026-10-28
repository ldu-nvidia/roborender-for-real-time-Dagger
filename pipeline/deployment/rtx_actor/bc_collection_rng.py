"""Preserve Raven collect-mode RNG without training a student on native RGB.

The pinned nonrecurrent collector discards the student's actions and incidental
weights. Its one full-batch learner nevertheless consumes a device randperm and
a discarded Normal sample. Keeping those draws preserves the subsequent reset
stream. This is collection RNG equivalence, not an optimizer-equivalence claim.
"""
from __future__ import annotations
import hashlib
import inspect

PROTOCOL = 'roborender.bc-collection-incidental-rng.v1'
POLICY_SHA256 = '86acfbd0cf610aec49e3a11e7049b18d4ca83d3bf28bcb63fc3484683c1cb239'
MODULE_SHA256 = 'f8d9b8a2a261ee0ca17306648bf2838d87947a310931ee30d38f74ed13c09965'
PPO_SHA256 = 'f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61'


def expected_receipt(batch_size, windows):
    return dict(protocol=PROTOCOL, policy_sha256=POLICY_SHA256, module_sha256=MODULE_SHA256,
                ppo_sha256=PPO_SHA256, original_learner_epochs=1, optimizer_steps=0,
                native_rgb_optimization=False, batch_size=batch_size,
                normal_shape=[batch_size, 81, 11], completed_windows=windows)


def validate_native(owner):
    """Reject unreviewed architectures, batching or stochastic transforms."""
    import torch
    from pathlib import Path
    policy = owner.policy_model
    args = owner.args
    if (type(policy).__name__ != 'VisionActorObjPred' or policy.is_recurrent
            or policy.num_actions != 11 or policy.max_rollout_history != 1
            or policy.running_mean_std is not None or policy.state_history_module is not None
            or policy.temporal_attention_module is not None
            or args.num_ppo_epochs != 0 or args.num_mini_batches != 1
            or args.local_batch_size != owner.env.num_envs
            or args.local_mini_batch_size != args.local_batch_size
            or args.per_device_train_batch_size != args.local_batch_size
            or args.num_micro_batches != 1 or owner.num_steps_per_env != 81
            or owner.vg_mode != 'collect' or owner.vg_rollout_action_source != 'teacher'):
        raise ValueError('RNG continuation requires pinned nonrecurrent full-batch teacher collection')
    paths = [(inspect.getsourcefile(type(policy)), POLICY_SHA256),
             (inspect.getsourcefile(type(policy.vision_module)), MODULE_SHA256)]
    from gr00t.rl.trl.trainer.ppo_trainer import TRLPPOTrainer
    paths.append((inspect.getsourcefile(TRLPPOTrainer), PPO_SHA256))
    for name, expected in paths:
        if name is None or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('incidental RNG reference implementation changed')
    if (any(isinstance(layer, torch.nn.modules.dropout._DropoutNd) and layer.p
            for layer in policy.modules()) or hasattr(policy, 'transform_train')
            or any(p.dtype != torch.float32 for p in policy.parameters())):
        raise ValueError('unreviewed dropout, augmentation or parameter dtype changes RNG')
    value=expected_receipt(int(owner.env.num_envs),0)
    value.pop('completed_windows')
    return value


def consume_window(batch_size, *, device):
    import torch
    if type(batch_size) is not int or not 1 <= batch_size <= 50:
        raise ValueError('bounded full environment population required')
    # Exactly the original train-loop order and shape/dtype/device. Mean/std
    # do not affect how many standard-normal variates Torch consumes.
    torch.randperm(batch_size, device=device)
    with torch.no_grad():
        mean = torch.zeros((batch_size, 81, 11), dtype=torch.float32, device=device)
        torch.distributions.Normal(mean, torch.ones_like(mean), validate_args=False).sample()
