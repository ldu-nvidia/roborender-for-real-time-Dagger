# Clean BC → DAgger collection plan — October 20, 2026 (Pacific)

**October24 override:** the clean BC collection/continuation has produced the
selected epoch116 baseline; do not recollect BC for the current DAgger request.
The current DAgger protocol supersedes this plan's held-out DAgger population:
generated-only training,100 accepted updates/round, no held-out collection or
MSE stopping, update100 handoff and the same corrected rendered295 test after
each round. BC dataset provenance/validation is unchanged. See the
[current contract](END_TO_END_PIPELINE_CONTRACT.md).

## Decision and execution status

This supersedes October15 legacy reuse and the October16 2,000+560 extension
for the next experiment. User direction: remove previous problematic collection
data to avoid confusion, and **start with fresh BC training-data collection**.
Do not reuse old trajectories, rendered derivatives or exposed student/AdamW
checkpoints in the new experiment. This is an approved plan, not a completion
receipt, native qualification, or authorization to release compute.

Correction source: commit `9178f7a` plus the active `LeggedRobotBase` coverage
correction in source release `clean-bc-reset-20261020-v2`, shared
`roborender.initial-reset-completion.v1`. CPU regression tests and bounded native
BC B6 qualification passed; other widths and DAgger phase switches remain gated.
The v1 native retry rejected missing reset evidence before stepping; it did not
qualify the active collection path. Do not reuse that failed attempt.
Read [the reset fix and exposure](INITIAL_RESET_FIX_AND_EXPOSURE_20261020.md).
The prior teacher262/295 (88.81%) result qualifies its earlier corrected
evaluation protocol, not execution of this new whole-pipeline source build.

Fresh training collection started with campaign
`bc-campaign-clean-reset-train-20261020` on Leo GPU0, serial B6 and40GiB host cap.
Twenty-five sessions cover2,552 training chunks; the final eight chunks in the
B1 tail require separate actual-width qualification. No qualification data is
admitted. Campaign SHA256:
`6dc0b611ec4e21022b3fcd7bec20f837e3fb7269235cb3c9b88bc5c09e8f7b05`.
The full target remains2,560; these are launch counts, not completed data counts.

The two-cohort-per-process launch subsequently failed its presave memory guard
in its second cohort; no OOM/swap occurred. Its first54 selected chunks and all
failure evidence are preserved but not admitted from that failed session.
Recovery is a fresh `bc-leo-clean-single-20261020` identity with exactly one
cohort per process, the unchanged sampling plan/seed/quotas, and unchanged
qualified native source/memory guards. Guarded scheduling/admission now rejects
multi-cohort lifetimes because the capacity receipt did not qualify them.
This new B6 schedule has49 sessions for2,552 chunks; the eight-chunk B1 tail
still requires its own actual-width admission.

Collection and offline rendering may overlap. The stream renderer transfers only
completed source sessions with matching corrected-reset/execution receipts, then
renders independent clips on the retained H200s while RTX collects the next
session. This does not change selection, seeds, chunk order, render parameters
or final whole-population admission. No active writer or qualification output
can enter the queue; neither stage releases compute on completion/failure.

## BC populations and sampling

| Role | Selected population | Use |
|---|---|---|
| Training | 2,560 ×81-row chunks =207,360 frames | Teacher-supervised11-D action MSE |
| Validation | 320 ×81-row chunks =25,920 frames | Full held-out MSE; stopping and checkpoint selection |
| Test | Same frozen295 closed-loop cases | Stage-end success; no optimizer or offline test-video pool |

Freeze the295-motion inventory, asset hashes, seeds, quotas and chunk positions
before collection. Use `pipeline/bc_collection/sampling_plan.py`'s all-fresh
`build_offline_plan` / matched-exposure recipe, not the legacy-reuse admission:

- Training:200 motions ×9 chunks +95 motions ×8 chunks. Motion assignment/order
  is seeded and fixed; keep each motion's complete native chunk sequence.
- Validation:295 motions ×1 chunk +25 preselected extra quotas. Separate seeded
  cohorts; preselect held-out chunk ordinals across each motion's8–9-chunk
  training exposure span rather than taking only startup prefixes.
- Preserve chronological frames and complete within-episode81-row BC chunks.
  Discard incomplete terminal tails. Never split neighboring frames into roles
  or select chunks using reward, success or loss.
- The selected quota is2,880 chunks; native collection may emit more while
  reaching held-out ordinals, handling resets and synchronizing lanes. Budget
  disk/time for this additional output, not only selected chunks.
- Record native reset samples, globally scoped episode identities, motion and
  source/reset hashes, seeds, chronological chunk ordinals and render lineage.
  Reject known trajectory/source/reference leakage; different names or seeds
  alone do not prove distinct physical initial states.

Capture synchronized three-view controls, actor/teacher/Homie observations,
privileged teacher26 action, object state and identity/outcome metadata under
the existing data contract. Only teacher action drives BC collection.
Render both newly admitted splits with the pinned independent three-view
50-step teacher recipe (CFG5, seed42, max_chain1, no cross-clip references).
Keep state/action rows paired unchanged. Rerendering old conditions is not repair.

This is in-population validation across the same motions, not unseen-motion
generalization. Matching exposure support does not guarantee identical task-stage
frequency; report actual coverage and reset/phase distribution limitations.

## BC learning, DAgger and final tests

Start BC from the approved immutable pre-BC bootstrap with a fresh optimizer.
Training MSE optimizes; complete held-out validation MSE governs stopping and
absolute-best checkpoint selection. The collection decision does not silently
change learning hyperparameters or stopping budgets; freeze those in the new
run's configuration. Log train/validation/test metrics and source/data identity.

For teacher probabilities75%,50%,25%,0% in order:

1. Inherit the preceding clean stage's selected policy and AdamW state.
2. Collect a fresh independent320-chunk validation population covering all295
   motions with that round's frozen parent policy and beta; hold it fixed for
   the entire round and exclude it from optimization.
3. Collect fresh training trajectories online with separate seeds. Teacher
   labels every state, regardless of selected behavior. Retain81 sequential
   transitions per lane per update; DAgger windows may span real episode ends.
4. Use validation MSE for stopping/selection, then run the fixed success test.

After BC and every DAgger round use exactly the same295-case panel, seeds,
motion assets and cohort layout with the corrected evaluator, selected student
and Homie only. No teacher takeover or optimizer. Retain the pinned ray-traced
three-view observation mode and original success predicate. Success does not
choose checkpoints or change patience. Because this panel has informed research
decisions, regard it as a development benchmark; a final generalization claim
needs an untouched test. Do not replace the agreed panel silently.

## Safe rollout and old-data removal

1. Inventory exact old raw/rendered/held-out/training collection targets and
   active consumers on shared storage and RTX. Preserve compact manifests,
   lineage and deletion records; never relabel old results as corrected.
2. Delete only confirmed problematic collection payloads under the user's
   instruction. Do not delete broad workspace/data roots, models/checkpoints,
   teacher/Homie/renderers, scene/motion assets, anchors, source or295 test cases.
   Resolve aliases/hardlinks and stop if ownership or live-use scope is unclear.
3. Freeze corrected source/overlay/config; qualify native first-step robot/object
   continuity, ordinary resets, BC collection, DAgger train/heldout and warm
   train→test→train switching before admitting fresh production data.
4. Begin fresh BC training collection in a new namespace, then validation and
   rendering/admission. Verify source/reset receipts, counts, pairing, finiteness,
   split separation and deterministic deep samples without repeated full rescans.
5. Reuse hardware/assets, not old trajectory data. Target two Leo RTX GPUs and
   four H200 workers; choose batch sizes from actual whole-pipeline capacity
   evidence for this frozen build. Do not inherit historical B2/28/64 defaults.

Keep all requested/allocated H200 compute unless explicitly authorized to
release that exact resource. A workload stop or data cleanup is not release
authorization. Do not patch live frozen simulators or replay uncertain windows.
