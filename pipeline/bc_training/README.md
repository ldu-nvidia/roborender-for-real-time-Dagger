# BC training

**October26 cyclic-loader correction (new fixed-final runs):** sampling v2
matches Raven's persistent sorted-file cursor, including wrap inside a full
batch. It never drops a tail or resets the cursor at an epoch boundary.
For non-divisible pools, a reporting epoch means `ceil(N/B)` full updates,
not exactly one exposure per chunk. MSE averages those actual training rows.
Checkpoints bind the cursor to completed optimizer updates; prefetch does not
advance committed progress. Resume rejects old/different sampling semantics.
Legacy validation paths retain their historical divisibility gate. CPU
differential/resume tests pass; current-source live qualification is separate.

**October26 current diagnostic:** all old BC datasets/checkpoints remain retired;
the later contract overrides historical merged2880 defaults below. An explicit
fresh smoke path (`--fresh-continuous-pool`, `--fresh-render-plan`,
`--fresh-rendered-root`, supplied together) trains from64 newly collected and
rendered complete clips, with two full32×81 updates from the initial bootstrap.
It never falls back to the retired selector or promotes a diagnostic checkpoint.
This live check passed, including policy/AdamW/RNG reload and W&B readback; see
`runtime/raven-dagger-alignment-20261026-7pYLxI/fresh-parent-evidence-v2.json`.
It is not a full BC epoch, converged model or downstream DAgger certification.

**Latest October25 explicit B32 request:** new `fixed_final` replay runs default
to `CHUNK_BATCH_SIZE=32` / `--chunk-batch-size 32`. This is one complete
**32×81 = 2,592-row** learning/BatchNorm population, never two B16 forwards or
gradient accumulation. The versioned32 batch contract has90 updates per full
2,880-chunk epoch. `legacy_validation` still defaults to16; admitted B16 replay
resumes and `fixed_final` direct runs require explicit `--chunk-batch-size 16`.
Frozen releases remain unchanged; strict resume rejects a16↔32 change. Explicit
epoch budgets and no-validation rules remain unchanged. B32 changes Raven's original B16
population and requires actual memory qualification; CPU tests alone do not
certify capacity.

`qualify_raven_bc32_lifecycle.py` is a separate, nonproduction two-update
diagnostic. It strictly reloads the immutable initial student checkpoint
`g1_3view_student_bootstrap_v1.pt` (SHA`720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a`),
not BC116 or retained constructor-random weights. Every initial parameter and
buffer is compared against that saved file. Each update deliberately samples16
original-train and16 former-validation chunks from the admitted merged pool;
these64 unique samples test both source roles, not the production sorted cursor.
It records finite gradients, full-population BN counters, AdamW state, exact
checkpoint/RNG reload, source/data/initial-weight hashes and actual memory.
Execution requires explicit `--execute`, a new isolated output, at least120GiB
free device memory, and a process allocator cap preserving16GiB device headroom.
It never retries, reduces the batch, evicts a renderer, launches production, or
releases compute. Dry-run metadata admission makes no CUDA calls.

**October25 dataset promotion:** the user retired held-out BC validation. The
new `bc_train_union_splits` storage-registry entry contains a versioned training
index with **2,560 + 320 = 2,880 chunks / 233,280 frames / 90 B32 updates per
epoch**. It references the existing clean rendered files; no payload is moved,
copied, recollected or rendered again. The original2560/320 indexes, source
roles, test descriptors and BC116 checkpoint remain immutable history.
The former320 chunks are no longer a held-out set for this training lineage.

`train_union.py` publishes/revalidates the distinct
`roborender.bc-train-union-admission.v1`: exact union, original production/render
admission, original physical dataset root, unique paths/IDs, complete81-frame
chunks, file existence/sizes and unchanged test provenance. This is a metadata
and existing-receipt audit, not a fresh hash/decode of every payload. Only
`fixed_final` accepts it; no legacy/qualification bypass or old-checkpoint
exact-resume waiver. The empty `validation-retired.json` is an explicit
compatibility index, not a zero-sample validation measurement.

The standard launcher now selects this union by default for `fixed_final`,
through the central Cosmos `bc_train_union_splits` registry entry and its
admission-bound physical dataset root. It resolves `train.json`,
`validation-retired.json`, and `data-admission.json` beneath
`/workspace/project/roborender-artifacts/datasets/split_manifests/clean_reset_merged2880_20261025`.
To select another admitted pool, provide the complete `DATASET_ROOT`,
`TRAIN_SPLIT`, `VALIDATION_SPLIT`, and `DATA_ADMISSION_JSON` set (environment
variables or their full CLI options); partial overrides fail to avoid mixing
datasets. `legacy_validation` retains its explicitly historical defaults.
Supply an explicit epoch budget. This merge does not start training, replace
BC116, or authorize silently resuming BC116 on changed data. The protected test
reference in the old admission is historical (ray-traced); it is not an
evaluation launch permit. New tests still require current generated-only
scaled-evaluation admission.

**October25 versioned offline-student lifecycle:** fresh `fixed_final` runs now
default to `--student-lifecycle raven_offline_replay_v1`. Keep generated/rendered
RGB for both training and testing. Before each full configured B×81 learning
forward, this mode calls the pinned student's rollout method81 times in train/no-grad mode,
including its discarded Normal samples, then clears its observation buffer and
applies Raven's device-side lane permutation. BatchNorm therefore records82
updates, not1; activation recomputation restores buffers and RNG without adding
an extra persistent update. This adds81 student forwards per optimization step
and may materially increase training time; no new GPU timing is certified.

The mode requires the existing finite `obs_gt_obj_pos` payload (at least3
coordinates; use the first3, accepting both the current3-D and original wider
layouts), unfreezes the auxiliary object head after strict bootstrap loading,
and computes action MSE +0×object MSE. The object head receives zero gradients
and AdamW state; with zero weight decay and fresh zero moments its weights stay
unchanged. Std remains frozen. Saved optimizer metadata records the actual
trainable set and PyTorch version. Missing auxiliary observations, nonfinite
losses, active dropout, RMS, recurrence, longer history, or shuffled chunk
membership fail closed. No teacher/Homie inference or native simulation is run
to replay the saved student inputs and labels.

Historical `legacy_validation`/BC116 retains `direct_supervised_v1`, frozen
object head, its original batching identity, and one BN update. Explicit direct
mode is also available for fixed-final compatibility experiments. A checkpoint
cannot resume across lifecycle/batching/optimizer identities; BC116 cannot be
silently migrated. The hash-pinned CPU differential test compares two updates
against original Raven methods: gradients, weights, BN, RNG, lane permutation,
and auxiliary optimizer state. It uses the **same explicit PyTorch AdamW on both
sides**, not the uncertified historical Transformers/Accelerate installation.
Raven declares TRL0.17.0 and0.19.0 in different dependency files; exact external
optimizer implementation, teacher/environment RNG and GPU numeric parity remain
uncertified. The original82-versus1 BN difference is not proof of a policy-quality
cause or a promised improvement. See `test/pipeline/test_bc_raven_offline_semantics.py`
and the historical BC source re-audit in `pipeline/RAVEN_EQUIVALENCE_AUDIT.md`.
This source change launches no run and changes no frozen release or selected BC116.

**October24 current protocol:** new runs default to `--training-protocol
fixed_final`. Supply an explicit positive `--max-epochs` budget; there is no
inferred production budget. Train only, skip held-out and train-evaluation
passes, and select the last fully completed epoch for success testing. Save
policy/AdamW/RNG/counters and publish the hash-bound `training-complete.json`
only after durable checkpoint publication and successful logging finalization.
An interrupted partial epoch cannot be selected as complete. Resume requires
the same protocol/budget and exact data/batching/optimizer/RNG identity.
If final logging failed after the complete checkpoint, `--finalize-completed-run
/absolute/run/path` reconciles the saved publication intent and same W&B run
without CUDA, training or checkpoint unpickling. Missing intent, changed pins,
or unconfirmed logging fail closed; they do not authorize replaying training.

BC116 remains the user-selected baseline; this change does not retrain or
reinterpret it. Existing validation files/provenance are retained for data
admission, but not loaded for MSE passes. The validation/patience sections below
describe historical runs and require explicit `--training-protocol
legacy_validation` compatibility opt-in; they are not new-run defaults.
Do not invoke the old validation-based smoke/production controller as the new
warm paired-testing workflow. See the active production contract's phase barrier.

Owns supervised fitting with Raven's action loss, optimizer, full batch and
sorted/cyclic loading, retaining the requested three-view policy. The qualified
render recipe is `raven_three_view_independent_v2`, not the historical
autoregressive pool. The launcher defaults to `REQUIRE_PRODUCTION_DATA_ADMISSION=1`
and the registered merged train-only index described above. Historical
`splits_independent_v2` defaults apply only to `legacy_validation`.
Missing corrected inputs fail closed; data selection is not launch admission.
The manifest pins content, disjoint episode/render lineage, and recorded motion
and simulator-seed coverage. Historical diagnostics require explicit legacy
paths and `REQUIRE_PRODUCTION_DATA_ADMISSION=0` (warning emitted); that is not
production certification. `ROBORENDER_ASSET_ROOT` binds an absolute shared asset
root when executing a frozen source release, keeping heavy runtime/data/checkpoints
outside version control. Training reads only its
index; test arrays never enter optimization or checkpoint selection.

## Historical collection admission (preserved beneath the new train union)

The October 15 successor plan is 2,560 training and 320 MSE-validation chunks,
covering all 295 admitted motions with matched exposure. It is distinct from
the 20-chunk qualification dataset and from legacy rendered pools. The small
qualification pool must never be labeled representative production data.

`production_protocol.py` declares the purpose and required populations;
`assemble_production_protocol.py` binds completed collection/render evidence;
`plan_success_test.py` prepares the independently seeded, equal-motion
295-case closed-loop test. These management tools do not themselves establish
that collection, rendering or evaluation ran. Production admission requires
their actual execution receipts and pinned source/model/render identities.

If completed sessions used different independently qualified memory limits,
the assembler emits protocol `v4`: each immutable launch/host/session chain is
bound to its own capacity-receipt SHA, actual environment width, and exact cgroup
limit. Host, concurrency, reserve, collector source, and scientific sampling
remain common. Earlier receipts are never relabeled to a later limit. Admission
also checks each session's observed memory/headroom and zero OOM/swap evidence.
Single-profile campaigns retain `v3`. A mixed campaign requires a freshly pinned
validator release supporting `v4`; never patch a running/frozen collector or
render release to bypass an older validator's rejection.

For overlapping collection/rendering, `render_execution.finish_streamed` binds
the original completed cohort worker plans to the final pool. It preserves each
worker's original asset identity, requires identical renderer/model bindings,
rejects duplicate/missing/changed outputs, and reads only receipts/file identities.
It does not fabricate a merged worker or rerender the completed dataset. Final
activation still checks paired labels/state and samples RGB before admission.

There is no offline test-video split in this successor data protocol. Historically, training used
the training pool, stopping and checkpoint selection use only held-out MSE,
and the selected checkpoint is then measured in native closed-loop success
testing. Test outcomes must not feed checkpoint selection, patience or physics
changes. The validation stopping rule is an explicit user-requested extension
to Raven's fixed-budget training, not a claim of byte-identical scheduling.

## Data path

For each indexed 81-frame chunk the loader reads only the learner fields:

- `rgb_gen__{head_color,hand_left,hand_right}` are tiled vertically into the
  `(324,192,3)` policy observation.
- `obs_actor_obs` supplies the 113-dimensional non-privileged student state.
- `gt_actions[..., :11]` supplies the privileged-teacher target for the
  student's 11 action outputs.

Renderer-only depth, mask, and robot-control arrays stay sealed in the NPZ but
are not decompressed by training.

## Model and objective

The trainable student is the existing DAgger-compatible policy. Training starts
from the immutable `g1_3view_student_bootstrap_v1.pt`, whose ResNet-18 contains
the pinned ImageNet initialization. Runtime construction sets the ResNet
constructor's `pretrained=false` to prevent a second download, then strictly
loads the pinned bootstrap before optimization. ResNet features are
projected to 128 dimensions and concatenated
with actor state, followed by the SiLU MLP `[512,256,128]` and an 11-D action
head. Both ResNet and MLP are optimized with uniform elementwise MSE against
the first 11 dimensions of `gt_actions`. Value, entropy, object-prediction, and
locomotion-controller losses are absent; the offline trainer does not load the
robot teacher or locomotion controllers.

## Historical validation and saturation (explicit compatibility only)

The native PyTorch trainer evaluates the complete admitted validation index at
initialization and after every full pass over the admitted training index.
Population counts come from those manifests, not historical 1600/200 counts.
Validation runs in evaluation/inference mode and never updates
parameters or optimizer state. It never initializes Isaac, Vulkan, or either
RoboRender model.

Every validation point is saved immutably under
`validation_checkpoints/epoch_NNNN_step_NNNNNN.pt`; this includes the epoch-zero
baseline. `last.pt` aliases the newest validation checkpoint and
`best_validation.pt` always aliases the absolute lowest validation-MSE checkpoint,
even when an improvement is too small to reset patience. A result counts as a
meaningful improvement only when MSE decreases by more than
`max(1e-6, 0.005 * patience_best_mse)`, measured against the last meaningful
minimum. Several small cumulative improvements can therefore reset patience.
Training stops as saturated after eight
consecutive full-validation checks without such an improvement, but never
before ten epochs. One hundred epochs is the hard safety ceiling; the normal
terminal condition is the validation plateau, not reaching that ceiling. The
test arrays are never opened by this stage. These validation/stopping rules are
an explicit protocol extension: Raven's original launches used fixed budgets.

`validation_state.json` records every validation result, the exact stopping
state, and the selected checkpoint. These W&B series are emitted:

- `train/mse`
- `validation/mse` and `validation/best_mse`
- `validation/mse_action_00` through `validation/mse_action_10`
- validation epoch, duration, patience count, improvement, and saturation

The W&B project is `dagger`, group `BC`, and the production run display name is
`BC`. Local output directories remain timestamped so checkpoints are never
overwritten by a later run.

- `train_raven_bc_h200.sh` launches `train_bc_h200.py` directly in the qualified
  H200 Python runtime; it does not launch the GR00T simulator entry point.
- Each loader item is one synchronized 81-frame chunk. Every training batch
  contains exactly 16 chunks, or 1,296 supervised frames, matching Raven's
  complete `num_envs=16, num_steps_per_env=81, num_mini_batches=1` policy
  forward and optimizer update. The ResNet forward uses activation
  recomputation and an index-safe, sample-only max-pool split; neither operation
  splits BatchNorm's 1,296-frame reduction population.
- Training chunks are lexically sorted and consumed cyclically without shuffle,
  matching Raven's offline cursor. `--experimental-shuffle` is a visibly
  non-default research change. Validation chunks are never shuffled or sampled.

The training split must be exactly divisible by 16. The launcher and trainer
both fail closed otherwise: they never pad, duplicate, or silently drop a
partial final batch, because any of those choices would change either the
training distribution or SyncBatchNorm population. `drop_last=True` is retained
as a defensive DataLoader invariant, but drops zero chunks in every accepted
run. Validation remains unchanged: it runs in `eval()`/inference mode over the
entire validation split, so its smaller loader batch is not a training
normalization population.

Scientific parameters remain Raven's: fresh ImageNet-pretrained trainable
ResNet18; vision feature dimension 128; SiLU MLP `[512,256,128]`; actor state
plus vision input; uniform L2/MSE on the first 11 dimensions of the 26-D
teacher action; coefficient 1; actor LR `2e-4`; constant learning rate; value,
entropy and object-prediction losses absent; frozen object-prediction head and
action standard deviation; global gradient norm capped at `1.0`.

The learner receives only the train index. Validation is read exclusively by
the parameter-free selection loop, and the test index is not referenced by
the launcher or trainer.

## Historical validation-based resume training

October22 user-directed continuation: `--stop-training-epoch-mse-below 0.015
--no-epoch-cap` requires both the complete epoch's **training-mode**,
element-weighted average MSE strictly below0.015 and the unchanged eight-check
validation plateau. It does not use one minibatch or a frozen eval-mode training
pass. `--no-epoch-cap` is genuinely unbounded; explicit user stop, nonfinite
loss/runtime failures and resource safety guards remain active. Every epoch
still validates, saves last/best checkpoints and logs `train/mse_epoch` plus
`train/epoch_target_met`. The absolute minimum validation MSE still selects
`best_validation.pt`. Training longer can overfit and does not guarantee higher
closed-loop success. The older optional evaluation-mode goal below remains a
different, mutually exclusive experiment. No downstream DAgger is auto-started
by this offline BC continuation.

Optional October19 fitting-target experiment: `--stop-training-eval-mse-below 0.02`
requires **both** the existing validation plateau and full-training-set frozen
evaluation-mode MSE strictly below0.02 before early stopping. The unchanged
`--max-epochs` ceiling can still stop without reaching the target. It measures
the training pool at resume/baseline and every epoch without gradients or BN
updates, using a separate non-shuffled loader. W&B `train/eval_mse` is distinct
from training-mode `train/mse` and `train/mse_step`; `validation/plateau` is
distinct from the conjunctive `validation/saturated`. Lowest held-out MSE still
selects `best_validation.pt`; training fit or final-test success never does.
This can extend training beyond overfitting onset and does not promise improved
success. Resume from an older checkpoint may opt into this explicit recorded
contract; subsequent resumes must retain the checkpoint's fitting-stop contract.

Pass `--resume-checkpoint` to continue in a new immutable output directory.
Resume strictly restores the policy, AdamW moments, epoch, global step, prior
validation history, best-checkpoint selection state, and—when available—the
Python, NumPy, PyTorch CPU/CUDA, and DataLoader-generator RNG states. It rejects
changed policy specifications, bootstrap provenance, train/validation split
hashes, or learning rates. `--max-epochs` is the final epoch ceiling, not the
number of additional epochs.

New checkpoints also seal the exact 16×81 batching contract, index-safe pool
installation count, and activation-checkpointing mode. Resume rejects a legacy
B=4 checkpoint or any changed batching contract; mixing their AdamW moments
would not be a faithful continuation. Sampling mode and ordered-path digest are
also pinned; an old checkpoint with unrecorded sampling is not silently changed
to sorted/cyclic mode. Start corrected independent-render BC
training from the immutable three-view bootstrap instead.

Every checkpoint written by the corrected trainer contains the complete RNG
snapshot and supports exact future resume under the same data and batching
contract. For example:

```bash
pipeline/bc_training/train_raven_bc_h200.sh \
  --resume-checkpoint checkpoints/students/BC/<corrected-run>/last.pt \
  --run-name bc_3view_corrected_resume \
  --max-epochs 100 \
  --wandb-run-name BC-corrected-resume
```
# Fresh diagnostic parent after dataset retirement

`qualify_raven_bc32_lifecycle --fresh-continuous-pool ... --fresh-render-plan ...
--fresh-rendered-root ...` admits new Raven-continuous teacher collection and
identity-bound offline50 rendered outputs. All three paths are required; there
is no fallback to retired data/checkpoints. The smoke uses the first64 unique
complete native emissions, no outcome filtering, for two full32x81 updates from
the saved bootstrap. This creates a diagnostic parent, not a production model
or evidence of convergence. Source/renderer/teacher pairing, optimizer and RNG
reload checks remain mandatory. Tests live under `test/pipeline/`.
