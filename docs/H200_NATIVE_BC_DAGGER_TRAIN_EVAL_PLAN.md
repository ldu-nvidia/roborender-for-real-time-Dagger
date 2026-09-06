# H200 Raven-reproduction execution plan

This document supersedes the earlier saturation/beta plan. The scientific
reference is Raven's supplied GR00T commit `94d7985`; total environment count is
the only intended scientific scale change.

1. Collect flat 81-frame chunks with the privileged teacher driving Isaac.
   Preserve Raven's exact three cameras, extrinsics, depth controls, robot-mask
   RGB, observations, 26-D labels, seed behavior, and chunk/reset semantics.
2. Render every chunk offline with the 50-step `step-5294` teacher at CFG 5,
   seed 42, the fixed Raven prompt, no RGB reference, and offline slot order
   left/right/head. Preserve supervision in the same learner NPZ.
3. Train three-view offline BC from the complete flat rendered dataset using
   Raven's ResNet18/MLP, 81-step single-minibatch MSE update, LR `2e-4`, fixed
   zero auxiliary losses, 50,000-batch maximum, and 200-batch save interval.
   There is no validation-MSE early stopping in the reproduction.
4. Launch four sequential online Image3F training jobs. Each restores the full
   previous checkpoint. Every job uses 81 live steps/update, teacher-driven
   action and label, 50% per-episode ray-traced observation assignment, a
   uniformly sampled anchor file whose frame index 0 remains fixed for an
   episode, seed 42, and Image3F 5-step/CFG-1/TeaCache-0.3 inference. Only
   current live storage is trained; no replay is saved or mixed.
5. Evaluate checkpoints with the offline-BC baseline saved `config.yaml`, native
   ray-traced observations, exactly 300 completed episodes, and closed-loop task
   success rate.

The active three-view path is Raven's smoke-verified extension. Her published
57.3% BC and 69.3% DAgger figures used one head view and cannot serve as exact
numerical expectations for this architecture.
