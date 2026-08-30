#!/usr/bin/env python3
"""Validate the real RoboRender GR00T student on one CUDA device.

This is a model-bound gate, not a synthetic throughput benchmark.  It proves
that the exact hash-pinned parent checkpoint can be restored without the RTX
trainer dependency stack, that its portable runtime identity matches, and that
the production observation/action adapter supports a finite forward/backward
optimizer step while the auxiliary heads remain frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.actor import runtime_state_dict_sha256  # noqa: E402
from round_dagger.artifacts import sha256_file  # noqa: E402
from round_dagger.gr00t_policy import (  # noqa: E402
    create_vision_actor_obj_pred,
    load_checkpoint_strict,
    load_policy_spec,
    vision_actor_obj_pred_adapter_factory,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256  # noqa: E402


REPORT_SCHEMA = "fast_roborender.round_dagger.real_gr00t_cuda_smoke.v1"


class ValidationError(RuntimeError):
    """The real-policy CUDA gate failed closed."""


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def validate(
    *, policy_spec_path: Path, checkpoint_path: Path, output: Path, batch_size: int
) -> dict[str, object]:
    import torch

    if output.exists() or output.is_symlink():
        raise ValidationError("output report already exists")
    if batch_size < 1 or batch_size > 16:
        raise ValidationError("batch size must be between 1 and 16")
    if not torch.cuda.is_available():
        raise ValidationError("CUDA is unavailable")

    spec = load_policy_spec(policy_spec_path)
    expected_checkpoint_sha256 = spec["checkpoint_sha256"]
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValidationError("checkpoint bytes differ from portable policy spec")

    torch.manual_seed(731)
    torch.cuda.manual_seed_all(731)
    device = torch.device("cuda:0")
    policy = create_vision_actor_obj_pred(spec)
    load_checkpoint_strict(
        policy,
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        map_location="cpu",
    )
    restored_state_sha256 = runtime_state_dict_sha256(policy.state_dict())
    if restored_state_sha256 != spec["policy_state_sha256"]:
        raise ValidationError("restored policy state identity mismatch")
    runtime_sha256 = runtime_policy_identity_sha256(
        restored_state_sha256, spec["homie_controller_sha256"]
    )
    if runtime_sha256 != spec["runtime_policy_sha256"]:
        raise ValidationError("restored runtime policy identity mismatch")

    policy.to(device)
    policy.train()
    adapter = vision_actor_obj_pred_adapter_factory(spec)
    actor_width = int(spec["input_contract"]["actor_obs_dim"])
    vision_shape = tuple(int(item) for item in spec["input_contract"]["vision_shape"])
    actor_obs = torch.randn((batch_size, actor_width), device=device)
    vision_obs = torch.randn((batch_size, *vision_shape), device=device)
    targets = torch.randn(
        (batch_size, int(spec["output_contract"]["policy_action_dim"])),
        device=device,
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValidationError("policy has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-5, weight_decay=0.0)
    actions = adapter.actions(
        policy, {"actor_obs": actor_obs, "vision_obs": vision_obs}
    )
    loss = torch.nn.functional.mse_loss(actions, targets)
    if not math.isfinite(float(loss.detach().cpu())):
        raise ValidationError("forward loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable_gradient_count = sum(parameter.grad is not None for parameter in trainable)
    if trainable_gradient_count != len(trainable):
        raise ValidationError("one or more trainable parameters lack gradients")
    if any(parameter.grad is not None for parameter in policy.obj_pred_mlp.parameters()):
        raise ValidationError("frozen object head received gradients")
    if policy.std.grad is not None:
        raise ValidationError("frozen action sigma received gradients")
    before_step_sha256 = runtime_state_dict_sha256(policy.state_dict())
    optimizer.step()
    after_step_sha256 = runtime_state_dict_sha256(policy.state_dict())
    if before_step_sha256 == after_step_sha256:
        raise ValidationError("optimizer step did not change policy parameters")
    torch.cuda.synchronize(device)

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "policy_spec_path": str(policy_spec_path.resolve()),
        "policy_spec_sha256": sha256_file(policy_spec_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "restored_policy_state_sha256": restored_state_sha256,
        "runtime_policy_sha256": runtime_sha256,
        "batch_size": batch_size,
        "action_shape": list(actions.shape),
        "loss": float(loss.detach().cpu()),
        "trainable_parameter_tensors": len(trainable),
        "trainable_gradient_tensors": trainable_gradient_count,
        "frozen_obj_pred_gradients": False,
        "frozen_std_gradient": False,
        "optimizer_changed_state": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
    }
    _atomic_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = validate(
        policy_spec_path=arguments.policy_spec,
        checkpoint_path=arguments.checkpoint,
        output=arguments.output,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
