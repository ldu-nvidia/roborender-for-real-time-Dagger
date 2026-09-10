# Success-rate evaluation certification

Date: 2026-09-09 UTC

Status: **PASS — Raven metric and closed-loop evaluation semantics**

This certification applies to the RoboRender three-view policy evaluation path.
It certifies that the evaluator measures the same task-success quantity as
Raven's implementation. It does not claim that the current three-view policy,
checkpoint, training data, or episode sample reproduces Raven's historical
single-view 57.3% experiment.

## Certified behavior

1. Evaluation is closed loop in Isaac: the policy action is applied to the
   environment and the next observation is produced from the resulting state.
2. Policy inference uses evaluation mode and no gradients.
3. The visual student supplies deterministic `action_mean` for 11 action
   dimensions. The deterministic 15-dimensional Homie action is concatenated,
   producing the 26-dimensional action sent to Isaac.
4. Success is sourced only from `infos["grasp_goal_reached"]`.
5. Success is Boolean-OR latched over every timestep of an episode, appended
   once when the episode terminates, and reduced as successful completed
   episodes divided by all completed episodes.
6. Reward and action MSE are diagnostics and do not define success.
7. Ray-traced evaluation does not call Image3F.
8. The three policy views are ordered `head_color`, `hand_left`,
   `hand_right`, resized to 108x192 per view, stacked vertically, converted to
   float [0,1], and normalized exactly once by ImageNet mean/std before the
   ResNet policy.
9. A candidate checkpoint is staged beside the selected baseline
   `config.yaml`, matching GR00T's checkpoint-local configuration loading
   behavior.

## Executed evidence

The corrected RTX evaluation completed exactly 300 episodes:

- successes: 49
- completed episodes: 300
- success rate: 0.16333333333333333
- Wilson 95% interval: [0.125806, 0.209373]
- mean episode length: 214.06 control steps
- observation: three-view ray-traced RGB; no Image3F
- policy: deterministic `action_mean`; no gradients
- checkpoint SHA-256:
  `4b7fbc3b10c5a3a6381fd6a89bd5ac779e045fcae8789769d2f0c084f4570a42`
- resolved configuration SHA-256:
  `557c3a64e7061553fcd689016b9a18f909ee6a41a173b2ad4bf55778ca0279e6`
- metrics artifact SHA-256:
  `b7306d0b56d324248115641a95936f232c45f8854ae7cb004f9afce1d6127dcd`
- no-step model/input attestation SHA-256:
  `93cb486ba136810a4619ec2d3aedee1a802a87e721f9f01919fa36182c620caa`

RTX evidence root:
`/mnt/data1/ldu/research/artifacts/roborender_bc_corrected_norm_b28_20260909T193611Z/`

Checkpoint loading was independently checked at 139/139 tensors,
11,615,725 parameters, zero missing/unexpected/unequal tensors, and maximum
absolute difference 0.

## Automated checks

The focused evaluation checks passed on 2026-09-09:

```text
7 passed, 10 deselected
```

They cover the Raven goal latch and denominator, success-only checkpoint
selection, fixed-panel expansion, evaluation launcher contract, and the H200
evaluation ledger. The relevant Python sources also passed bytecode
compilation.

The broader Raven byte-identity suite is not green: 18 checks passed and one
Raven tree-integrity check failed because the active checkout intentionally
contains the corrected three-view normalization and local BC split/validation
extensions. Consequently this certificate is semantic, not a claim of a clean
or byte-identical Raven checkout.

## Explicit exclusions

- Raven's historical 57.3% checkpoint was a different single-view policy and
  used 150 reported episodes. The current 16.33% result is not numerically
  comparable as a reproduction.
- The repository's sealed 256-case fixed validation panel is a local
  experimental-control extension. It uses Raven's success metric but was not
  Raven's historical headline sampling procedure.
- The exact historical checkpoint-local configuration and checkpoint bytes no
  longer survive in the audited artifacts, so exact historical environment
  and RNG equivalence cannot be certified.
- Varying `num_envs` changes parallel RNG consumption and the sampled episode
  sequence even though it does not change the success definition.

## Certification decision

The pipeline is certified for selecting and reporting checkpoints by Raven's
closed-loop task-success definition. It must be described as
**Raven-metric-conformant three-view evaluation**, not as an exact reproduction
of Raven's historical single-view success percentage.
