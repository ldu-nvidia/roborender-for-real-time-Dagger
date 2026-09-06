# Dataset root

This is the only RoboRender data root. The Raven-equivalent active datasets are
flat, not pre-split:

- `BC/isaac/chunk_*.npz`: sealed 81-frame teacher demonstrations containing
  three-view renderer controls, non-vision observations, and 26-D teacher labels.
- `BC/isaac/manifest.json`: immutable inventory, hashes, shapes, and rank/global
  environment lineage for those chunks.
- `BC/rendered/chunk_*.npz`: the same chunks with `rgb_gen` and all three
  `rgb_gen__<view>` arrays added by the 50-step offline teacher.

Raven's offline learner cycles the complete flat rendered dataset. Her DAgger
learner trains directly from each current 81-step live rollout and does not
write, reload, or mix a persisted round replay dataset. Consequently
`dagger_1` through `dagger_4`, `validation`, `test`, and `splits` are inactive
empty layout remnants of the superseded custom protocol, not inputs to any
canonical launcher.

Runtime plans, logs, previews, and renderer endpoint files belong under
`../runtime`, not here.
