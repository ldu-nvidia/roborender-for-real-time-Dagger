# Dataset root

This is the only RoboRender data root. Behavioral cloning uses one sealed pool,
not collection-batch or train/validation/test wrappers:

- `BC/isaac/bc_chunk_NNNNNN.npz`: sealed 81-frame teacher demonstrations containing
  three-view renderer controls, non-vision observations, and 26-D teacher labels.
- `BC/rendered/bc_chunk_NNNNNN.npz`: the same chunks with `rgb_gen` and all three
  `rgb_gen__<view>` arrays added by the 50-step offline teacher.
- `BC/manifest.json`: unified one-to-one inventory, hashes, original collection
  lineage, teacher identity, and renderer identity.
- `BC/SEALED.json`: exact 2,000-source/2,000-rendered pairing seal.
- `BC/splits/{train,validation,test}.json`: non-overlapping 80/10/10 indexes;
  files remain in the unified pool and are not duplicated.
- `BC/splits/manifest.json`: split seed, grouping/stratification contract,
  counts, hashes, and leakage checks.

The identical basename maps each Isaac archive to its rendered counterpart.
Batch, rank, host, seed, and timestamp are provenance fields in the manifest,
not parts of the canonical filename.

The BC launcher cycles only the indexed training subset. Raven's DAgger
learner trains directly from each current 81-step live rollout and does not
write, reload, or mix a persisted round replay dataset. Consequently
no inactive batch directories or duplicated split archives are retained.

Runtime plans, logs, previews, and renderer endpoint files belong under
`../runtime`, not here.
