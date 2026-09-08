# Raven data contract

## Collected Isaac chunk

Each `BC/isaac/bc_chunk_NNNNNN.npz` contains 81 synchronized frames. Required
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
loader reads only `BC/splits/train.json`, orders that fixed index once, and
cycles indefinitely. It refuses to enumerate the complete pool, so validation
and test chunks cannot enter optimization accidentally.

Isaac and rendered archives use the same contiguous canonical IDs
(`bc_chunk_000000` through `bc_chunk_001999`). Future persisted DAgger data, if
introduced, must use `dagger_rNN_chunk_NNNNNN`; batch, host, rank, timestamp,
and seed belong in the manifest rather than the filename.

## Split indexes

`BC/splits/{train,validation,test}.json` map canonical chunk IDs to both their
Isaac and rendered archives without moving or copying data. The deterministic
80/10/10 split is grouped by collection stream and stratified by source batch
and rank. No collection stream occurs in more than one split. The split seed,
source-manifest hash, per-file hashes, and validation checks live in
`BC/splits/manifest.json`.

## Online DAgger

No replay dataset is published by the active DAgger experiment. At each live
step, Image3F may replace RGB before the policy forward pass; the teacher labels
the current state, while the canonical beta selector chooses whether the
teacher or student action is executed. The current 81-step storage is optimized
once and discarded. The next update and next round collect fresh live states;
cross-round replay and held-out DAgger splits are not active.
