# Provenance and naming standard

This file governs artifacts associated with both canonical checkouts:
`/workspace/project/roborender` on H200 and `/home/ldu/roborender` on RTX.
The machine-readable layout record is `REPOSITORY_LAYOUT_MANIFEST.json`;
checkpoint and renderer identities are further pinned by
`../checkpoints/manifest.json` and `../renderer/manifests/registry.v1.json`.

## Names describe function

- Robot policy: `<robot>_privileged_action_teacher_<task>_stepNNNNNN.pt`.
- Student bootstrap: `<robot>_3view_student_bootstrap_vN.pt`.
- Student checkpoint: immutable, content-addressed candidate beneath the owning
  `checkpoints/students/<stage>/<run>/` experiment directory, registered by
  parent/run/iteration identity before handoff. Compatibility `last.pt` links
  are allowed only for an upstream loader that requires that name.
- Renderer adapter: `roborender_<offline|online>_<role>_<steps>_lora_stepNNNNN.safetensors`.
- TensorRT plan: `roborender_<role>_sm90_<precision>_<view-contract>.engine`.
- Student-checkpoint stages are `BC`, `dagger_1`, `dagger_2`, `dagger_3`, and
  `dagger_4`. The active rounds are sequential warm starts with student-action
  probabilities 25%, 50%, 75%, and 100%. Raven's recorded beta-1 behavior is
  retained as reference provenance, not used as the active experiment schedule.

Original filenames are provenance fields, never production identifiers.
Changing the architecture, view order, action schema, preprocessing, adapter,
foundation model, precision, or GPU target requires a new name and manifest.

## Required manifest fields

Every accepted binary records a stable ID, functional role, status, canonical
relative path, byte count, SHA-256, original source/path, and source revision.
A derived artifact additionally records all parent hashes and build/runtime
parameters. A dataset seal records source and code-contract hashes, complete
chunk/environment IDs, frame count, state/action shapes, view order, alignment
map, and payload inventory. The active BC pool remains flat while its 80/10/10
train/validation/test membership is stored as non-copying index manifests.

Paths are locators; hashes are identity. A path without a matching manifest and
digest must not be used by an active launcher.

## Publication lifecycle

1. Write to a unique `.partial` path in the destination filesystem.
2. Validate schema, shapes, counts, finite values, episode completeness, and
   frame/state/action alignment.
3. Compute byte counts and SHA-256 for every payload and parent dependency.
4. Fsync payloads and the containing directory.
5. Atomically rename to the final versioned path and publish a read-only
   manifest. An existing final name is an error, even if a retry appears valid.
6. Each distributed DAgger update registers an immutable candidate. At the
   configured boundary, the sealed 256-case closed-loop goal-reached rate—not
   validation MSE—selects the best checkpoint. Final test remains separately
   sealed and reporting-only.

Sealed artifacts are append-only. Corrections create a new version whose
manifest cites the superseded manifest; they never edit old bytes in place.

## Move, alias, and archive rules

Before a move, resolve the exact source, record its inode/size/hash, and reject
ambiguous matches. Prefer a same-filesystem metadata move. Rehash afterward;
only then update manifests and launchers. There must be one physical copy of a
large accepted artifact.

A compatibility symlink is allowed only for an unavoidable upstream hard-coded
path. It must be relative where practical, resolve inside the canonical project
asset tree, and be listed in the repository layout manifest. Active code must
use the canonical path. Superseded but evidentially useful artifacts move under
an explicit `archive/` root and receive a non-production status; transient or
incorrect outputs may be removed only after their identity and replacement are
verified.

## Isolation rules

`data/` contains datasets only, `checkpoints/` contains policy weights only,
and `renderer/` contains renderer weights, engines, contracts, and render
evidence only. Runtime environments, caches, logs, and run receipts belong in
`runtime/` or ignored container state. The active path uses one flat BC pool
with index-only splits and live, current-iteration DAgger storage. Historical
runtime receipts must not be promoted into source-controlled dataset trees.
