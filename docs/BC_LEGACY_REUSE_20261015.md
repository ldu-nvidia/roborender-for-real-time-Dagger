# BC legacy reuse: October 15, 2026

## Approved population and purpose

The user selected a reuse-first alternative to the separately preserved
2,560-train / 320-validation fresh-collection plan. This protocol retains all
2,000 historical raw Isaac chunks as **training only**, collects 320 new
held-out validation chunks, and freezes a separate 295-case native closed-loop
success test. It does not require collecting another 560 training chunks.

| Population | Count | Use |
|---|---:|---|
| Historical raw training chunks | 2,000 × 81 rows | 125 complete B16 optimizer batches per epoch |
| Fresh native validation chunks | 320 × 81 rows | Baseline and per-epoch MSE; checkpoint selection and patience |
| Independent native test cases | 295 episodes | Success after immutable MSE-based checkpoint selection |

The original dataset, RGB outputs, splits, manifests, checkpoints and earlier
results are preserved. Old validation/test roles are retired only within this
new versioned experiment; no historical result is rewritten. The new canonical
dataset directory is `data/BC/legacy_reuse_v1_20261015/`.

## What is reused, and what is not certified

The source manifest is `data/BC/manifests/autoregressive_v1.json`, SHA256
`67f2652b1d1a8ade14dd2e2a33d98032f3e6bf63ab8f6e2c62a63e19f8fa68ef`.
Its 2,000 unique raw source identities refer to the archived Isaac pool under
`data/BC/archived/pre_autoregressive_v1_20260913T235716Z/isaac/`.
Teacher labels, actor states and ordered frames remain paired and unchanged.
The historical teacher and both Homie model hashes match the current native
collection model identities.

All historical chunks lack explicit motion/episode/reset provenance; 490 also
lack a recorded simulator seed. These fields remain unknown. Current filesystem
order or filenames cannot reconstruct the historical unsorted motion inventory.
This protocol does **not** certify matched motion exposure, all-295-motion
training coverage, or exact reset nonoverlap with unknown historical resets.
Isaac-version differences are an additional disclosed distribution risk.

The old rendered pool has 974 chain-first and 1,026 inherited-reference chunks.
Chain-first outputs are compatibility candidates, not automatically admitted
independent renders. Unless a separately reviewed adoption path proves the
required identities, the supported execution rerenders every raw chunk with
the pinned three-view 50-step / CFG5 / seed42 / max-chain1 recipe. This reuses
the trajectories and labels; it does not recollect them. Old RGB is untouched.

## Fresh validation and final test

Freeze the 295-motion inventory, native source/assets, new process seeds and
selected complete-chunk ordinals before collection. Validation covers every
motion once, with 25 additional motion quotas selected deterministically before
execution. Select whole chronological chunks from a declared nine-complete-
chunk exposure span. The native teacher, reset sampler, simulator stepping and
81-row writer do not change. This is a broad target validation population, not
proof that its motion/phase frequencies match the unrecorded legacy population.

Only selected, complete chunks enter validation. Preserve native episode IDs,
actual reset receipts, outcomes, source hashes and collection-time assignments.
Check new process seeds against every known historical seed and the final-test
seeds. Check source/trajectory identities wherever recorded; distinct process
names or seeds alone do not prove distinct physical initial states.

The final test is a separately seeded, fixed one-case-per-motion native panel,
not the 200 old offline test videos. It is never used to choose checkpoints,
adjust patience, tune physics or add more training against its outcomes.

## Learning contract

Start from the immutable three-view ImageNet bootstrap, not a historical BC
checkpoint. Keep Raven's complete 16×81 forward, sorted cyclic loading,
uniform 11-action MSE, AdamW learning rate 0.0002 and gradient clipping at 1.0.
Randomized collection coverage is not random frame order or a new learner
shuffle algorithm. Preserve the actual 15-action Homie tail in source artifacts.

Measure the complete validation population before training and after each
epoch. Save every validation checkpoint and retain the absolute minimum MSE.
Use the existing declared minimum-epoch, patience, meaningful-improvement and
maximum-epoch limits. Train and validation are the only offline learner roles.
Run native success testing only after selection; never promise a success rate.

## Admission and execution status

Use a separate legacy-reuse schema. Do not disable production admission or
fabricate native cohort receipts to fit the all-fresh protocol. Admission must
bind the original 2,000 source identities, actual new render execution evidence,
strict fresh-validation collection evidence and the frozen success panel.
Planning alone does not authorize learning on an incomplete population.

Finalize each rendered population with the explicit `--manifest-only` option
to `pipeline/bc_rendering/activate_corrected_render_pool.py`. Use stratified
RGB sampling, not the default full RGB decode. This produces the validated
training-only or validation-only render manifest without inventing another
80/10/10 split. The legacy-reuse admission command alone publishes the combined
2,000/320 learner indexes after both populations and their execution receipts
are complete.

Current work is recorded under
`runtime/experiments/bc-legacy-reuse-20261015/`. Read its live receipts for actual
collection/render counts. This document records the approved protocol, not a
claim that collection, rendering, training or final testing has completed.
