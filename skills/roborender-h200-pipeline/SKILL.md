---
name: roborender-h200-pipeline
description: Resume, audit, or operate the canonical local-H200 RoboRender BC and four-round DAgger pipeline. Use for this repository's collection, rendering, training, validation, testing, asset provenance, or checkpoint bookkeeping.
---

# RoboRender H200 pipeline

Work only in `/workspace/project/roborender`. Read `README.md` first, then the
README for the relevant phase. Treat JSON manifests and executable contracts as
authoritative; treat old run logs only as immutable provenance.

## Fixed architecture

- Code: `pipeline/`; models: `checkpoints/`; renderer: `renderer/`; all datasets:
  `data/`; local environments/output receipts: `runtime/`.
- BC collection: flat `data/BC/isaac/chunk_*.npz` plus `manifest.json`.
- Offline rendered BC: flat `data/BC/rendered/chunk_*.npz`.
- DAgger storage is live for one 81-step update and is not persisted/replayed.
- Selected student stages: `checkpoints/students/{BC,dagger_1..4}`.
- Active contracts: `pipeline/bc_collection/raven_bc_collection_contract.v1.json`
  and `pipeline/dagger/raven_image3f_dagger_contract.v1.json`.

Do not create an alternate project/data root, a `dagger_5`, a campaign directory,
or RTX transfer machinery. All active compute is native to one eight-H200 SM90
node.

## Non-obvious contracts

The privileged robot teacher—not Isaac itself—produces ground-truth actions.
BC executes teacher actions. Raven's reproduced DAgger also executes and labels
with the teacher in every round (beta 1). Its 0.5 mix is the fraction of
environment episodes kept ray-traced, not the probability of teacher action.
Each update trains only on current live storage and each round warm-starts the
full prior checkpoint.

The 50-step video teacher renders complete BC sequences offline. Online Image3F
renders every DAgger observation from current depth/robot controls plus a fixed
photorealistic anchor triplet for that episode. Read
`renderer/contracts/CONDITIONING_CONTRACT.md` before touching either path.

Raven did not use validation-MSE saturation or a separate sealed test set. Use
the fixed 300-episode ray-traced closed-loop success evaluation with the saved
offline-BC baseline environment config.

## Provenance

Before using or moving a model, read `checkpoints/manifest.json` or
`renderer/manifests/registry.v1.json`. Verify size and SHA-256. Preserve one
physical copy; use compatibility symlinks only for unavoidable upstream paths.
Never overwrite a sealed binary, dataset, manifest, or selected checkpoint.
A changed contract creates a new version with parent hashes.

## Change workflow

Inspect both root and nested GR00T Git status and preserve unrelated changes.
After edits, run focused unit tests, Python compilation, shell syntax checks,
JSON parsing, manifest/hash verification, `git diff --check`, and the active
protocol validator. Do not claim a production gate from an old receipt.

Large datasets, environments, weights, ONNX, TensorRT engines, logs, and media
stay outside Git. Commit only source, manifests, and documentation after
reviewing staged paths for secrets and generated binaries.
