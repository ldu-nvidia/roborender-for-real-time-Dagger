# Raven data contract

## Collected Isaac chunk

Each flat `BC/isaac/chunk_*.npz` contains 81 synchronized frames. Required
three-view fields are native 480x832 quantized depth
`depth_rawq__{hand_left,hand_right,head_color}`, 240x416 metric depth
`depth_raw__*`, h5_v4 uint8 depth controls `depth_ctrl__*`, and textured
robot-on-black uint8 controls `rgb_robot__*`. The same chunk stores all
non-vision `obs_*` tensors, `(81,26)` `gt_actions`, `(81,)` `dones`, prompt,
local `env_id`, and Raven `seq`.

The privileged robot checkpoint produces `gt_actions`; Isaac produces dynamics
and observations. Raven emits a complete chunk before processing a done flag,
stores false dones in emitted chunks, discards partial episode tails, and uses
the original sequence discontinuity around reset. The seal adds global/rank
lineage without changing payload bytes.

## Offline-rendered chunk

The 50-step teacher preserves every input key and adds:

```text
rgb_gen                     (81,108,192,3) uint8  # head_color alias
rgb_gen__head_color         (81,108,192,3) uint8
rgb_gen__hand_left          (81,108,192,3) uint8
rgb_gen__hand_right         (81,108,192,3) uint8
```

Rendered frame `t`, state `t`, and teacher action `t` remain in one NPZ. The BC
loader sorts the flat chunk paths once, assigns a disjoint stride only when
using the optional H200 distributed topology, and cycles indefinitely.

## Online DAgger

No DAgger dataset is published in Raven's implementation. At each live step,
Image3F may replace RGB before the policy forward pass; the teacher action is
then both the executed action and the supervised label. The current 81-step
storage is optimized once and discarded. The next update and next launched
round collect fresh live state. Cross-round replay, saved proposed/executed
action rows, held-out splits, and beta bookkeeping are not active.
