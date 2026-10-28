# October 15: fresh BC and MSE-selected DAgger qualification

Evidence checkpoint: 2026-10-15 14:55 UTC. This report supersedes the metric and
data-readiness defaults of older smoke reports, while preserving their evidence.
It does not claim production policy quality, identical Isaac-version numerics,
or Raven's reported single-view success rate.

## What actually ran

Fresh Leo native teacher collection produced 20 selected 81-row chunks, followed
by 20 genuine independent three-view, 50-step H200 renders. The qualification pool
has 16 training / 2 validation / 2 historical offline-test chunks, only two motions and
known early-phase heldout bias. It is not admitted as representative production
data. The successor protocol omits unused offline-test video rendering.

BC performed three complete 16×81 updates. Train MSE decreased 0.694893→0.687502;
fixed validation MSE decreased 0.812882→0.806000. Strict resume reproduced the
baseline and restored all 70 AdamW parameter states, counters and RNG. A separate
bounded continuation stopped by validation patience at epoch 18; absolute-best
validation MSE 0.792346 occurred at epoch 15, while final training MSE was 0.421317.
This proves the distinction between lowest-MSE checkpoint selection and the
meaningful-improvement patience threshold, not generalization on broad data.

The epoch-15 selected checkpoint was subsequently evaluated once on the exact
same two native cases used for epoch3. Actual initial-state, reset/RNG and runtime
physics comparisons passed. Both checkpoints scored0/2: epoch15's first case
terminated for low object height at101 steps (epoch3 timed out at401), while
both ended the second case for the original throw-object predicate at3 steps.
Thus lower heldout MSE did not improve the measured two-case success count;
this is neither a population-quality estimate nor a reason to relax the cases
or physics. The run used GPU1, no H200, no optimizer and no augmentation, and
the policy was unchanged. Total511.885s comprised478.102s cold startup,
27.698s evaluation and5.580s cleanup. Minimum host headroom37.08GiB, zero owned
OOM/swap/limit events, clean authenticated owner shutdown. Evidence:
`runtime/production-audits/20261015-raven-faithful/bc-epoch15-success-diagnostic-20261015T142500Z/`.

The four-round campaign intentionally retained its frozen epoch-3 BC parent
`f49bddcc132cc721b2ab930a1e41106c9e562b377e22eafa66b1b6bc1d8a73ba`, not the
separate epoch-15 diagnostic checkpoint. Each round used two Leo actors with
two environments each, four genuine H200 Image3F/student workers, 81 steps per
lane and one full 4×81 optimizer update. It collected actual separately seeded
held-out parent-policy trajectories and tested only the MSE-selected checkpoint.

| Round | Teacher probability | Train MSE | Fixed heldout: parent → selected | Actual final success |
|---|---:|---:|---:|---:|
|1|75%|1.387049|1.514819→1.511681|0/2|
|2|50%|1.336889|1.953729→1.951578|0/2|
|3|25%|2.371965|4.221982→4.204662|0/2|
|4|0%|9.800781|5.459266→5.426325|0/2|

All rounds stopped at their explicit one-update smoke ceiling, not convergence.
The final checkpoint is
`66ebe59bcc1b2a53c48fb643bf167227bc6583258cc12a7d2a03957a507770cc`:
global/AdamW step 7, with the exact selected model+optimizer lineage from BC.
R4 behavior was 0 teacher / 324 student actions; teacher labels remained present.
Native final tests were forward-only, three-view raytraced, no augmentation,
no optimizer, and original pre-reset goal accounting. Test outcomes never
selected checkpoints. A 0/2 rate has a 95% Wilson interval approximately 0–65.76%:
it is not a useful population-quality estimate or evidence of improvement.

Both native owners retained one Kit initialization across phases and rounds.
Minimum measured host availability was 23.333 GiB; all owned swap/OOM/limit-pressure
counters zero. All four campaign H200 allocations were released. Frozen source,
controller, native, checkpoint and online logging receipts are preserved under:

`runtime/leo-smokes/dagger-leo-smoke-mse-b4-20261015-v5/`

The authenticated owners were then shut down at their clean IDLE boundary;
there is no simulator state claimed recoverable after shutdown.

## Logging and diagnostics

- BC: https://wandb.ai/nvidia/dagger/runs/6imjdkdr
- BC patience diagnostic: https://wandb.ai/nvidia/dagger/runs/es7onvkg
- BC epoch15 matched native test: https://wandb.ai/nvidia/dagger/runs/bc-epoch15-success-20261015
- R1: https://wandb.ai/nvidia/dagger/runs/beta-d1069934580d7b287cced4cf
- R2: https://wandb.ai/nvidia/dagger/runs/beta-6262381d1b899294bc6ef8e7
- R3: https://wandb.ai/nvidia/dagger/runs/beta-6e48edb68e0009393cc468c9
- R4: https://wandb.ai/nvidia/dagger/runs/beta-e521dc5aa8420c421f841434

Online summary readback is part of completed-round receipts. The successor
reporting patch distinguishes held-out MSE validation from final success testing
and clears stale previous-round display fields. The completed runs received a
documented status-only summary correction, verified through API readback; numeric
summaries and configurations stayed unchanged, and no history was written. This
does not retroactively change the frozen source used by those runs.

The large cross-round MSE is consistent with a weak, three-update parent and
changing target distributions: a constant BC-training-mean predictor has
held-out errors 1.505856/1.952845/4.206884/5.429523 for R1–R4. R4 training's same
baseline is 9.775572 versus measured 9.800781. This descriptive comparison does
not establish that distribution shift alone explains historical production
results. All 20 BatchNorm tracking counters match optimizer steps; no duplicate
BN updates were found. Bootstrap ResNet18 weights match official ImageNet V1.

A separately owned one-H200, forward-only diagnostic compared the exact same
train/validation arrays and checkpoint in disposable train/eval modes. At
epoch 3, validation MSE was 0.805371/0.806000; at epoch 15, 0.793213/0.792346.
Thus mode switching does not explain this smoke's large heldout-distribution
loss jump. Epoch-15 training data did show 0.489170 in train mode versus 0.626144
in eval mode, consistent with differing BatchNorm statistics after very few
updates. No moments were changed in a saved policy, no optimizer ran, and no
test data was read. The allocation was released after 12.61 seconds of actual
diagnostic work. Preserve Raven's normalization rather than tuning it against
the final success panel. Evidence: `bc-forward-mode-results-v1.json` under the
dated production-audit directory.

The matched teacher/BC native diagnostic gave 1/2 versus 0/2. One shared reset caused
both to terminate at 3 steps near the original 0.1 m object-to-hand throw threshold.
Preserve/report that reset sensitivity; do not relax physics or remove hard
test cases to inflate success.

## Production gates, not completed claims

The separate original-buffer BC qualification now passed **B=1 and B=6** on one
Leo GPU, with a 43 GiB cgroup cap and an 8 GiB host reserve. B1 executed 243 native
steps and three full saves; B6 executed 405 steps and five full saves, with every
lane reaching the required second complete chunk. Measured peaks were 27.04 and
33.22 GiB; minimum host availability was 33.69 and 21.93 GiB. Owned swap, OOM and
limit-pressure counters stayed zero. This supports the bounded single-collector
profile, not two concurrent B6 BC collectors or a mathematical maximum.

The first B6 attempt correctly failed its four-window quota because one motion
had emitted only one complete chunk. A fresh, explicitly eight-window-bounded
retry preserved the same motion/seed selection and passed after five windows;
the difficult motion was not removed. B1's initial metadata publication failed
on a root-owned output directory after native work had completed. Its original
failure remains recorded; metadata-only ownership/publication recovery was
attested, and the shared guarded launcher now precreates exact cohort output
directories as the invoking user. Production must use that qualified guarded
launcher, not the historical plain session launcher.

B6 capacity receipt SHA256:
`6ad121a312a14c5e2ba753d101cbf840a864c1a701485bea204f40da9477dd3b`.
B1 capacity receipt SHA256:
`4168bf5e30df4c027282009c564d62f9cfdb1bbfeb108079f00a193c0b48170e`.
These bind actual native execution; CPU test counts do not substitute for them.

The reviewed successor specifies 2,560 train + 320 MSE-validation chunks across all
295 eligible motion assets, matched-exposure ordinal sampling, and an independent
295-case equal-motion success panel. The native geometry check now passed all 40
variants in seven B6 scenes (two explicit padding lanes), one Kit initialization,
seven vector steps and no optimizer/writer calls. It verified the declared table
metadata, USD extents/colliders, finite three-view captures and exact teacher/Homie
action tails. Its initial diagnostic-only USD metadata parser was corrected to
separate `userDocBrief` from Raven table records; no simulation source changed.
The fresh successful result SHA256 is
`da36bc3e8ad8e346cc6487faa1c5118373194849b7a40fa8598819b6e979a735`.
Runtime was 540.09 seconds including startup, with 103.28 seconds inside the
seven-scene native diagnostic. This is an asset/geometry check, not full-episode
dynamics equivalence or policy success evidence.

The first full-plan production collection session then completed: eight selected
training chunks and the separately seeded validation cohort's ninth complete
chunk (ordinal8), with 17 total emissions retained on Leo. It performed 2,106
native vector steps (810 train +1,296 validation), one Kit initialization and
two scenes. Wall time was963.974s, including536.070s native session time; the
warm scene rebuild took7.322s. Peak cgroup use26.83GiB, minimum host availability
33.13GiB, zero OOM/swap/limit events. The owner exited cleanly; no second session
was launched.

All nine selected payloads (306,505,141 bytes) and their small native/execution
provenance reached CPU shared storage, with one destination hash verification
per file. Source files and unselected audit emissions remain on Leo. The prefix
is explicitly **incomplete and not admitted for production training**:
`runtime/production-audits/20261015-raven-faithful/bc-full295-prod1-prefix-session0000/`.
The collection-only W&B run is
https://wandb.ai/nvidia/dagger/runs/bc-full295-prefix-20261015 ; online summary
readback verified actual counts/timing and optimizer_steps=0. It must not be
confused with the completed small-pool BC/DAgger optimizer runs above.
An additional bounded diagnostic opened only actor/action ZIP members in these
nine files: all had81 finite rows with actor113 and teacher26 shapes, and no
cross-role exact actor113-plus-student-target11 duplicate was found. No image or
depth array was opened, no full payload was rehashed, and no full-pool admission,
near-duplicate or physical-state-independence claim follows from that check.

The remaining data gates are the other25 production sessions, complete
provenance, actual renderer identity and strict split admission. Capacity is constrained by Leo's 62 GiB host
RAM; abundant GPU VRAM does not authorize two large BC buffer owners.

The bounded BC session controller and protocol/success handoff are management
tools. Only verified completed sessions are resumable; ACTIVE/UNKNOWN native
work is not replayed. The four-cohort session bound is conservative, not proof
that Kit fails after four cohorts. A longer warm lifetime needs its own test.

A metadata-only storage audit found that maximum declared emission counts could
exceed Leo's root-disk capacity despite modest observed compression. The future
v2 controller now requires explicit per-session storage budgets, substantial
reserve and one-second free-space/identity checks. It stops only its own runner
on a breach and leaves incomplete work non-replayable. This is CPU-regression
tested protection, not a guaranteed compression bound or a guarantee against
unrelated writers exhausting disk between checks. The completed v1 session's
state is immutable: continuation must use a fresh v2 campaign over the remaining
sessions and preserve the nine verified chunks, not rewrite/adopt old ACTIVE
state or replay the first session.
A fresh25-session continuation is staged plan-only under
`runtime/production-audits/20261015-raven-faithful/bc-full295-prod1-continuation-v2-20261015/`.
Its remaining launch rows exactly match the original schedule after session0;
the existing9 plus future2,871 selected chunks still form the unchanged2,880
plan. The prior READY prefix and receipt chain were independently reconstructed.
No continuation state, simulator, renderer or learner was started.

Additional source-only hardening accepts the new two-role BC plan without
flattening its separately pinned grouped success panel. MSE learner admission
now rejects missing or mismatched AdamW/global-step/scheduler state before model
or CUDA initialization. Requested learning rate must equal the saved rate for
exact continuation; an intentional change needs an explicit new protocol.
Already-loaded moment shapes/dtypes and scalar counters are checked, without
an extra checkpoint hash or full moment scan. The actual BC and four selected
smoke checkpoints pass this guard with 70 optimizer states and steps 3 through 7.

Full-population DAgger still needs a versioned long-lived cohort executor with
round-wide heldout selection and checkpoint/optimizer continuity across actual
cohort boundaries. Calling the existing whole-round operation once per 81 steps
would incorrectly reset native history. No full 295-motion DAgger production
or greater-than-70% success result is claimed by this qualification.

## Final source regression scope

The consolidated working source passed1,694 CPU tests in146.51s, compilation,
`git diff --check`, the skill validator and bounded source/secret scans. These
are distinct from the genuine native/H200 evidence above. No Git commit/push
is implied by a source-only snapshot. Production continuation must use its exact
frozen source/config identities; do not hot-patch a native owner or treat a
matching historical Git HEAD as execution provenance.
