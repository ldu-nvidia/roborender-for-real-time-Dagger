# Raven parity and deviation audit

Reference: `qingh097/fast_roborender`, supplied `videogen-pipeline` snapshot,
with the three patch files that produce GR00T commit
`94d79851d9c471b1210be81c75116d9bea835a4d`.

This document separates three claims that must not be conflated:

- **source-exact**: the executed Raven file/config is byte-identical;
- **semantic wrapper**: the wrapper delegates to hash-attested Raven functions
  while changing deployment topology; and
- **experiment extension**: a requested behavior differs from Raven's recorded
  experiment and is explicitly versioned here.

The current system is Raven-derived. Matching the goal predicate is not proof
of matching the whole evaluator or training algorithm. The October25 re-audit
below identifies unresolved implementation and experiment differences; older
"semantic wrapper" descriptions are scoped claims, not blanket certification.

**Current authority:** use [the active contract](../docs/END_TO_END_PIPELINE_CONTRACT.md)
and the dated October26 corrections below. This audit retains historical findings:
the merged2,880-chunk pool and BC116 lineage described below were subsequently
retired and must not be reused. Later cyclic-loader, full reset-inventory,
native selection, learner-permutation and shared-curriculum fixes supersede
earlier descriptions of those defects; each still needs its stated live evidence.
Current execution and unfinished gates are tracked in
`runtime/raven-parity-overnight-20261026-t6oMpN/HANDOFF.md`.

### October26 newly confirmed warm-event boundary defect

Raven constructs a fresh task/event manager. Our warm same-scene reuse restored
the environment step clock but left `_reset_term_last_triggered_step_id` and
`_reset_term_last_triggered_once` from the preceding phase. With the native
20-step reset-event throttle, a new phase at clock0 can therefore skip visual
randomization. The actual pinned IsaacLab3 `EventManager.apply` method reproduces
this on CPU; the running container's manager source SHA is
`d90c82254a950baa1fbec336557cf4a6c50a2d2b05d61e094a3da50802c55ddc`.
`ColdPhaseState` now snapshots/restores the actual cold reset-counter tensors,
preserving tensor identities and RNG. Unknown startup/interval/class-event state
fails closed. Ordinary episode/window boundaries are unchanged.

The live v4 repeat had identical initial robot/object/reset/observation hashes
and depth controls, but different robot-RGB controls in all32 lanes;18 episode
lengths differed while all32 success flags agreed. That is supporting evidence,
not proof that this single defect accounts for every pixel/trajectory difference.
The frozen v4 owners were not patched. New-source integrated live verification
is pending; CPU proof must not be promoted to a native determinism guarantee.

## October25 end-to-end comparison: controlling scope

### October26 live integration boundary — incomplete

Both16-lane V5 actors collected the first81-step window with generated input,
but learner service inherited visibility of all8H200s. Its single-device guard
rejected the request before constructing the model or updating weights; both
native drivers then terminated. This was infrastructure placement, not an
algorithm change or OOM. The fix pins the learner alone; renderer/gateway
ownership remains8-way. Focused tests and a real retry of the exact saved32×81
batch passed (one update,MSE1.35105264,78AdamW states,~49.6GiB minimum free).
The failed original scope/ledger is preserved. This isolated learner proof
cannot certify live continuity or warm train→test→train→test; those remain open.
Fresh V5b same-source owners use a new diagnostic scope, never native replay.
The new live scope subsequently completed its first actual32×81 update:
MSE1.36879897,78AdamW states preserved, actual lane permutation and saved-RNG
continuation receipts,~49.6GiB free H200 memory. Both owners continued into
window2. This closes the placement failure at that boundary, not the full
four-update continuity or warm-cycle certification.

V5b then failed at the second81-step teacher cleanup. This exposed another
real wrapper discrepancy: pinned `Actor.clear_rollout()` clears buffers and
deletes `distribution`; deterministic `act_inference()` does not recreate that
attribute. Raven's `TRLDistillTrainer._rollout_step` catches exceptions from its
optional teacher init/clear hooks; our native lifecycle propagated them.
The corrected wrapper now follows that exception boundary (with visible
warnings), without swallowing student/environment errors or resetting physics.
The new regression executes the pinned Actor lifecycle/inference and Raven
wrapper for five81-step windows; the second-clear failure is separately reproduced.
Thirty-eight focused tests pass; frozenV6 has116 boundary/regression tests and
25 parent-import/device-placement tests passing. The preceding fullCPU run
passed5603 tests before this latest fix. These are not interchangeable live
certificates. Failed V5b physical state is excluded, its one completed update
preserved, and a fresh V6 diagnostic is required before continuity certification.

The user explicitly retained **generated/rendered student inputs for both
training and testing**, while asking to match Raven's remaining logic. Never
restore the native-RGB learner-storage bug or ray-traced student fallback in
the name of parity. Preserve the corrected robot/object initialization and
terminal-goal export. The agreed two-RTX scale, three views, four mixed-policy
rounds,100 updates per round and295-case test are separately recorded experiment
settings, not claims that Raven ran that experiment.

The comparison covers actual collection/render receipts and the pinned original
source, not just similarly named functions. Remaining limits include the missing
original offline-render server implementation, the historical BC BatchNorm
replay mismatch, training curriculum handoff/population topology, and evaluation
reset/sampling differences. CPU tests do not certify live Isaac6 physics,
numerical renderer identity, scaled capacity, or future success rates. New
source fixes cannot retroactively change the data or BC116 weights.

The requested additive BC merge is complete: **2,560 original training +320
former validation =2,880 chunks /233,280 frames**. The new index is
`/workspace/project/roborender-artifacts/datasets/split_manifests/clean_reset_merged2880_20261025/train.json`.
`data-admission.json` binds the unchanged original admission and sources;
`validation-retired.json` contains zero items and is compatibility metadata,
not a validation metric. Original files, old split memberships and the
historical test descriptor remain unchanged. The union is accepted only by
the no-validation `fixed_final` training path. It does not admit evaluation,
choose a BC epoch budget, launch training, or permit an old-checkpoint strict
resume under a new data identity.

## October25 DAgger continuity correction and re-comparison

The previous two-window cohort scheduler was **not equivalent** to Raven's
physical rollout lifecycle. Its forced scene replacement after162 steps could
discard unfinished trajectories. New continuous-round plan/config/queue schemas
remove that replacement; historical schemas and evidence are not rewritten.

Reference tree: `runtime/raven_gr00t_collection_94d7985`, commit
`94d79851d9c471b1210be81c75116d9bea835a4d`. Key source locations are
`gr00t/rl/trl/trainer/ppo_trainer.py` (`train`, `_rollout_step`),
`distill_trainer.py` (`_rollout_step`, `_get_rollout_data`,
`_get_mb_rollout_data`, `_compute_dagger_bc_loss`),
`distill_trainer_videogen.py` (`policy_step`, `_image_dagger_relabel`),
`actor_critic_modules.py` (`init_rollout`), and the environment's
`reset_from_dataset_wsppt.py` / `loco_manip_base.py` reset and step functions.

| Logic | Raven reference | New continuous implementation |
|---|---|---|
| Initial physical state | One initial reset; randomized episode/stage clocks; reference-frame poses | Same reset/sampling rules once per round; corrected robot/object reset retained, not identical RNG draws or initial states |
| Motion assignment | Lane index modulo available motions; resets stay in that motion | Explicit inventory-ordered32 motions, same lane mapping in all four rounds; no295-motion rotation claim |
| Physical trajectory | Carry returned observations across81-step windows; native done/timeout resets lanes | One accepted command per RTX for100 windows; no reset at81,162 or the qualification gate |
| Model rollout buffers | Student and teacher begin/clear every81-step window | Same per-window lifecycle, separate from physical episode state |
| Supervision | Teacher labels every visited state, including student-driven states | Same11-D action target; Homie supplies the15-D control tail |
| Learning batch | Current81 steps × all environments; full-batch action MSE, then next rollout |32×81 fresh rows, one full-batch update, immutable checkpoint barrier before next window; no replay |
| End of run | Finite update budget can end with unfinished physical episodes |100 accepted updates; final policy+AdamW handoff, not a promise that every trajectory succeeded or reached task end |

There is **scoped control/data-flow parity, not literal algorithmic or numeric
identity**. Explicit retained differences are B32 versus Raven's B16, three
views, beta75/50/25/0 rather than the supplied teacher-only launchers,100 updates,
generated-only inputs and final rendered testing. The original default
`enforce_teacher_rollout` forces a teacher subset before the probability draw;
we preserve the user's explicit per-decision beta rather than copy that bias.
The original Isaac5 versus this Isaac6 port is not numerically certified.

Two original defects are deliberately not reproduced: the supplied
`image_dagger` path uses generated RGB for action prediction on generated-assigned
lanes but stores their unchanged native RGB for learning; and window-local `dones` can miss an anchor
refresh when termination is exactly at step81. Our learner stores the same
generated observation used for prediction, and persistent episode identities
refresh anchors on every actual reset. The corrected robot/object reset stays
mandatory in collection and testing. These exceptions prohibit a claim of
byte-identical execution, even though retaining them is intentional.

CPU tests in `test/pipeline/test_dagger_raven_continuity.py` execute extracted,
SHA-pinned original rollout methods for five consecutive windows and compare
the new driver/window-runner trace. They also run100 windows/8,100 vector steps
with one physical initialization and a termination precisely at step81, and
reproduce the original observation-storage mismatch. Simulator/model/renderer/
transport dependencies are CPU doubles: this is not live physics, throughput,
capacity, full distributed-optimizer or success-rate certification.
The October25 focused regression run passed282 tests covering continuous/legacy
plans, actor/provider gates, resource monitoring, learner batching and native
entrypoint/window semantics. A further27 native-production integration tests
passed after correcting outdated rendered-test and THP-stub fixtures, retaining
their guard/reset/cleanup assertions (309 distinct tests total). This is a
focused suite, not an all-repository green claim. No new native workload was
deployed by this correction.

Native admission must bind both current-source owners, four accepted prefix
updates and actual learner/assignment/resource/W&B receipts while both owners
wait without resetting. Continuation to100 uses those same commands/PIDs;
terminal quiescence cannot be substituted by fabricated receipts. The provider
may reconnect to surviving work, never restore a lost simulator from weights.

**October24 deliberate protocol change:** new DAgger runs use100% Image3F5
student observations, not Raven's50/50 native/generated observation mixture.
Teacher behavior remains75/50/25/0% across four rounds. Each round performs100
full B32×81 updates, no held-out-MSE validation/patience, and tests/passes its
final update100 policy+AdamW. BC keeps its offline teacher50 renders; all student
success tests use online Image3F5 with the corrected same295 cases. Renderer
conditioning can still consume Isaac RGB/depth. This removes the native-RGB
policy-input path, but does not claim Image3F5 equals offline teacher50 or that
success must improve. Historical parity and score evidence below is unchanged.

## Current Leo protocol and measured scope (2026-10-15)

The current authority is the October 15 override in
[the pipeline contract](../docs/END_TO_END_PIPELINE_CONTRACT.md). The latest
complete native smoke used Leo GPUs **0 and 1**, B=2 per actor, four approved
H200 workers, full 4×81 learner updates and one update per mixed-policy round.
All four rounds completed actual held-out MSE validation, selected-model/AdamW
handoff and independent two-case success tests. This is functional evidence,
not broad policy-quality certification. See
[the measured report](deployment/QUALIFICATION_20261015_MSE.md).

Current offline rendering follows the supplied three-view launcher with
**max_chain=1**. The legacy six-clip autoregressive pool is not equivalent to
that launcher and contains cross-split reference dependencies. New production
data must pass the corrected admission protocol; old files are preserved, not
renamed into the new pool.

The requested validation extension selects on **held-out action MSE**, then
tests the selected policy for closed-loop success. The success panel must not
select checkpoints or stopping thresholds. There is no cross-iteration/round
replay in the faithful path. Source-exact teacher/writer calls do not establish
identical numerical physics after the Isaac 6 port.

## Historical Beta deployment override (2026-10-12)

The Alpha topology/replay descriptions below are historical. The versioned
Beta contract uses **two RTX GPUs (physical 1 and 2), six environments each,
four reserved H200s, v3 transport, and current-window-only learning without
replay**. Each update preserves one full 12×81 normalization population.
The native runtime is the pinned Isaac 6 port; numerical equivalence to the
old Isaac 5/Alpha experiment is not certified.

Production remains 50 updates per round and a fixed 256-case B5 success panel.
`roborender.cross_gpu_dagger_smoke.beta.v1` separately declares three full
updates per round and ten completed B5 success cases. This smoke exercises all
four betas and checkpoint handoffs, but cannot establish policy quality or
replace full-panel evaluation. No partial 256-case probe may be labeled a
complete production evaluation.

## Stage matrix

| Stage | Preserved Raven behavior | Local difference | Classification |
|---|---|---|---|
| BC collection | privileged teacher drives Isaac; 81-row emission; three cameras; 480×832 capture; 240×416 `h5_v4`; synchronized 26-D labels | process topology, output path, seeds/ranks, and scaled environment/collection count | source-exact inputs with scale/deployment overrides |
| Offline rendering | teacher LoRA step 5294; 50 denoising steps; CFG 5; seed 42; supplied three-view launcher uses independent clips with max_chain=1; offline left/right/head slots; 108×192 output | direct local multi-GPU worker replaces the server wrapper | parameter-equivalent semantic wrapper; legacy six-clip pool excluded |
| BC training | Raven three-view config; vertical head/left/right; ImageNet ResNet18; feature 128; SiLU MLP 512/256/128; LR 2e-4; MSE on 11 action dimensions; one full 16×81 normalization/update population | sealed local train/validation/test indexes, activation recomputation, index-safe sample-only max-pool, and checkpoint selection logic | three-view/local-control extension preserving full-batch statistics |
| Online observation | per-episode 50% ray-traced vs Image3F assignment; fixed anchor through episode; `h5_v4`; Image3F 5 steps, CFG 1, TeaCache 0.3, untiled VAE | current qualified path has four H200 workers and v3 binary transport | semantic wrapper plus deployment extension |
| Teacher/step | Raven privileged teacher + Homie; teacher labels every visited state; one vector `env.step` after action selection | exactly-once cross-host vector barrier | semantic wrapper |
| DAgger behavior | per-decision selection and current-window learning | teacher betas .75/.50/.25/0 instead of Raven's recorded beta 1 | explicit experiment extension |
| DAgger data | 81 live rows and one full-environment-batch update; current-window-only data | latest native smoke B=4; future cohort widths require actual resource and continuity qualification | scale/deployment extension, no replay |
| MSE validation | no original Raven early-stopping counterpart | independently seeded fixed parent-policy heldout population; absolute-lowest MSE selection and separate patience threshold | explicit requested statistical extension |
| Final success test | OR-latched `infos["grasp_goal_reached"]`; deterministic closed-loop policy+Homie; completed-episode denominator; no gradients | independent fixed panel after MSE selection; two actual smoke cases, 295-case production panel planned | experiment-control extension, not selection feedback |
| Final evaluation | ray-traced closed loop and completed-episode success rate | three-view checkpoint is not Raven's published single-view checkpoint | Raven-metric-conformant, not result-equivalent |

## Deliberate experiment extensions

### Three-view policy

The policy consumes `head_color`, `hand_left`, and `hand_right`, vertically
stacked at 108×192 per view. Raven smoke-tested this later extension. The
historical 57.3% and 69.3% results came from a single-head-view policy. The
preserved experiment uses teacher behavior (`beta=1`), regardless of a run's
BC/DAgger label; those labels do not establish that mixed-policy DAgger improved
success. Treat the numbers as historical context, not a numerical acceptance
target for this three-view model and the requested mixed-policy experiment.

### Action-mixture schedule

Raven's preserved config records teacher behavior (`beta=1`) in every reported
round. The active project schedule is the user's canonical DAgger experiment:

| Round | Teacher probability | Student probability |
|---:|---:|---:|
| 1 | .75 | .25 |
| 2 | .50 | .50 |
| 3 | .25 | .75 |
| 4 | 0 | 1.00 |

The teacher still labels every state. `raytraced_mix_prob=.5` controls the
observation source per episode and is unrelated to the behavior-action beta.

### Distributed topology

Raven's loop is split without moving ownership of the irreversible operation:

- Leo RTX GPUs 0 and 1 each own a persistent vectorized Isaac process, teacher
  inference, episode reset state, and their own `env.step([B,26])`.
- Four allocated H200 workers own Image3F and frozen student inference; the
  learner uses admitted compute for its full-batch update at the barrier.
- CPU access services orchestrate trajectory publication, registry and evaluation
  ledgers. H200 allocations are not retained just to host control processes.

The current v3 boundary uses bound binary tensors, stable idempotency identities,
an all-lane prepare barrier, and durable batched commit acknowledgements.
This changes transport and persistence, not teacher targets, rendered policy
observations, selected actions, or learner tensors.

### Held-out validation and independent success panel

Raven's success definition is preserved. The current user-requested protocol
uses held-out MSE for stopping and checkpoint selection, followed by a separate
success test. Validation trajectories remain fixed within a round and are
collected under its frozen parent policy. Different rounds have different
occupancy distributions, so their MSE values are not directly interchangeable.
The former 256-case success-selected schedule is historical, not current.

## Fixed data and model contracts

| Area | Value |
|---|---|
| BC/DAgger rollout window | 81 sequential rows per environment |
| Latest measured DAgger topology | B=2 on each of Leo RTX GPUs 0 and 1; not a proof of maximum capacity |
| Policy views | head, left hand, right hand; vertical; 108×192 each |
| Teacher action | 11-D privileged manipulation + 15-D Homie = 26-D label |
| Student output/loss | 11-D action mean; uniform L2/MSE to teacher `[0:11]` |
| BC optimizer | LR 2e-4, constant schedule, one minibatch |
| Online Image3F | adapter `rgb030-depth030-step-5000`; 5 steps; CFG 1; TeaCache .3 |
| Offline teacher | LoRA step 5294; 50 steps; CFG 5; seed 42; independent three-view clips, max_chain=1 |
| BC update population | exactly 16 chunks × 81 frames in one policy/BatchNorm population |
| Latest measured DAgger update population | exactly 4 chunks × 81 frames in one policy/BatchNorm population |
| Replay | none in the current faithful path; historical experimental replay configs are not enabled |
| Selection metric | independent held-out action MSE; success is measured only after selection |

## Camera contracts

The student order is always `head_color`, `hand_left`, `hand_right`. Renderer
slots differ by stage:

| Renderer | `ext1` | `ext2` | `wrist` |
|---|---|---|---|
| 50-step offline teacher | `hand_left` | `hand_right` | `head_color` |
| 5-step online Image3F | `head_color` | `hand_left` | `hand_right` |

Changing or unifying these mappings would be a scientific error.

## Pinned reference identities

| File | SHA-256 |
|---|---|
| `wsdpt_videogen_collect.yaml` | `c483a32b5a1810dd7abcedfb425b86c939d45e02a0ea43d448f8b0b226f21e5c` |
| `wsdpt_videogen_offline_bc_3view.yaml` | `eb2002a99d5dbdefb8ae58a8aaebc8b150cd406702f53a901841ad47e1a0529e` |
| `wsdpt_image_dagger_3view.yaml` | `787c461bb96f9e7f2116322c457b0c944ca12c124094b7f08afe9a3cb2a50b6f` |
| `distill_trainer_videogen.py` | `cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b` |
| `ppo_trainer.py` | `f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61` |
| `image3f_frame_client.py` | `2b5154001066ac6a1d1409016f7972d51bd8b12f702ce0b1bcac9c00a73c943a` |
| Raven `offline_render_dataset.py` reference | `4be283df7820d64628e9116d1a34055558677f04c454b8259165567177a1e31c` |
| Raven `image3f_server.py` reference | `d5d4631a7d3ff8b66aa4972140e3252b3cc2fc2498ef87a48fbd89ec60698963` |

The collection and Image3F validators check these identities before a new
deployment. Model and engine identities are in `checkpoints/manifest.json` and
`renderer/manifests/registry.v1.json`.

## Historical evaluation evidence and certification limits

The source checks cover Raven's metric and closed-loop semantics:
deterministic 11-D student action plus Homie, policy observation normalization
exactly once, OR-latched goal reached, and completed-episode denominator. The
historically recorded three-view BC evaluation was 49/300 (16.33%, Wilson 95% interval
12.58–20.94%). It is not comparable to Raven's historical single-view 57.3%
because the view topology, checkpoint, and sampled environment sequence differ.

See `evaluation/SUCCESS_RATE_EVALUATION_CERTIFICATION.md` for evidence and
explicit exclusions.

The immutable three-view comparison suite uses identical scenario IDs for the
BC checkpoint, each DAgger checkpoint, and the privileged-teacher oracle, with
separate ray-traced-only and Image3F-only student ablations. The current wrist
camera calibration manifest is intentionally provisional; native capture,
visual review, and runtime-extrinsic hashes are still required before a live
evaluation can be called fully certified.

## Audit rule

Any change to views, camera mapping, depth conversion, teacher correction,
action dimensions, observation-source assignment, renderer parameters, loss,
replay policy, success latch, or beta schedule requires a new versioned
contract and an entry here. Hardware paths, process placement, telemetry, and
architecture-local TensorRT compilation may vary only when their artifact
identity and effect are recorded.

## Privileged-teacher diagnostic split (2026-10-17)

The teacher-only positive control must distinguish two original Raven paths.
`distill_trainer_obj_pred_homie_api.py` cancels cumulative base velocity in
stages 1–3 and supplies `gt_actions`. Original
`ppo_trainer_homie_api.py` instead takes `policy.rollout(...).action_mean`,
concatenates deterministic Homie, and supplies only `actions`. The native
delta-action environment then suppresses applied base velocity while retaining
different cumulative/last-action feedback. Those conventions are not equivalent.

The opt-in `teacher_action_mode=raven_ppo` is diagnostic-only and teacher-only.
It preserves the original discarded sampling RNG call, frozen 226-D teacher RMS,
11-D teacher plus 15-D Homie construction, and absence of `gt_actions`.
Default BC/DAgger distillation is unchanged. Mode-bound receipts prevent reusing
results across the two conventions. The effective lexical teacher feature order
puts `delta_actions` at `[44:55]`, including velocity feedback `[44:47]`.

The initial matched diagnostic retains the frozen panel's seeds/motions/cases
and 8-second deadline. It records at most 24 steps of numeric state, commands,
joint speeds and contact signals, including terminal state before auto-reset.
Its failure-enriched case subset is **not** a population success-rate estimate.
Initial-state equality and native trajectories, not code similarity alone, are
required to attribute failures to the control convention. Stage-0 object
separation precedes the stage-1–3 correction and needs a separate contact/reset
investigation. Isaac 6 versus the original Isaac 5.1 remains an unproven numeric
physics-equivalence boundary. No success threshold or test case is relaxed.

## BC re-audit 2026-10-25

**Later October25 implementation:** new BC collection now has a separate
continuous plan/request/launcher/ingester; the measured old pool and adapter
identity below remain historical, not the new defaults. See
[`continuous_plan.py`](bc_collection/continuous_plan.py),
[`run_continuous.py`](bc_collection/run_continuous.py) and the
[current collection contract](../docs/END_TO_END_PIPELINE_CONTRACT.md).

The new path removes per-motion quotas, selected-ordinal filtering, validation
roles and short cohort restarts. It uses one shared vector scene, full ordered
motion inventory, fixed active lane prefix, once-only native initial clocks,
natural done/timeout resets, preserved curriculum and an explicit81-step-window
budget. The writer remains the pinned original; every complete emission is
retained. The adapter observes one initial `reset_all` and rejects a second;
completion requires normal return at the exact budget, no pending transitions
or emissions, and native/provenance agreement on unflushed partial tails.

The full inventory stays loaded even for unassigned motions, preserving the
reset sampler's per-inventory resampling behavior. Current source supports one
scene with B≤40 and at leastB unique eligible motions; it explicitly rejects
unsupported repeated modulo mappings instead of silently rotating motions.
Reference defaultB32 is a configuration, not a measured live capacity claim.
Two independentB16 collectors are not substituted for Raven's shared curriculum.
Original default50,000 windows is not evidence of the historical executed budget.

Reliability additions are durable per-window receipts, a bounded-memory chunk
index, actual launch/resource provenance and all-emissions pool verification.
No native replay, partial-tail padding, success-based selection, source hotpatch
or compute release is introduced. Qualification output cannot enter a production
pool. CPU tests execute the original writer across multiple windows and test
normal completion, terminal boundaries, early return, duplicate initialization
and mid-window failures. Native capacity/lifecycle qualification remains pending.

Explicit exceptions remain corrected initial refresh, the Isaac6 port and
zero-optimizer teacher collection. Raven's incidental native-image learner work
is omitted under the generated-only training requirement; RNG consumption and
bitwise physics are therefore not claimed identical. The supplied reference
already includes the three native conditioning views. These collection changes
do not certify downstream renderer/trainer numerical equivalence.

This comparison uses the clean local reference
`/workspace/project/roborender/runtime/raven_gr00t_collection_94d7985`, verified
at commit `94d79851d9c471b1210be81c75116d9bea835a4d`, and the actual clean BC
collection receipts. It does not establish numeric Isaac5/Isaac6 equivalence.
The reference collector, PPO trainer and collection YAML retain the SHA256
identities in the table above. Current
[`isaac6_bc_production.py`](deployment/rtx_actor/isaac6_bc_production.py)
and the executed `clean-bc-reset-20261020-v2` copy have identical SHA256
`3c64b6d04fd91d7bdf7ad2f137279c58f09b804a7382aced183cd33556517d31`.
The [executed production protocol](../runtime/clean-bc-20261020/prod-10-21/production-protocol.json)
binds collection source release
`983441add76361994d1f574aae5231e6a77380e9cffd7acede2bb9f675d771c9`,
IsaacSim6.0.0, IsaacLab commit
`28a37cecdd433c22d9eabd6a5954add9f13a8951`, and the teacher/Homie asset hashes.
Frozen historical artifacts retain their original roles and identities.

### Actual population and finite collection boundaries

The [sealed source pool](../data/BC/clean_single_20261020/manifest.json) contains
2,880 selected chunks. A read-only metadata audit checked the SHA256 of all100
referenced cohort manifests and read all2,880 selected provenance sidecars.
Every cohort reports complete teacher-only collection, zero optimizer updates,
randomized initial episode clocks and the corrected initial-reset receipt.
The [data admission](../data/BC/clean_single_20261020/splits/data-admission.json)
is production-admitted; the earlier metadata-only completion observation is
not the admission receipt. This audit did not rescan every image/tensor payload.

| Measured collection property | Original training role | Original validation role |
|---|---:|---:|
| Selected81-row chunks |2,560|320|
| Distinct motions |295|295|
| Per-motion selected quotas |200×9 +95×8|270×1 +25×2|
| Cohorts / independent process seeds |50 /50|50 /50|
| Cohort widths |49×B6 +1×B1|49×B6 +1×B1|
| Complete native chunks emitted |2,870|2,329|
| Complete overcollection chunks excluded |310|2,009|
| Partial rows discarded at native episode ends |57,192|47,045|
| Partial rows remaining at final quota stop |10,686|10,708|
| Selected chunks with true terminal outcome at row81 |10|1|

Thus native execution emitted5,199 complete chunks to select2,880. All100
cohort lifetimes ended with some unfinished partial rows. Training cohorts
ran891–1,215 vector steps; validation cohorts ran486–1,134. A quota stop is
neither an episode-completion guarantee nor a success guarantee. Selected
chunks are not independent episodes: there are1,229 distinct selected training
episode IDs and315 validation episode IDs, with multiple chunks from some
episodes. Initial episode counters actually range1–399 in training and1–397
in validation. The numbers above describe the historical collection roles;
later authorized reuse of those files does not change what was collected.

The original supplied
[launcher](vendor/raven-fast-roborender-reference/scripts/data/collect_data.sbatch)
defaults toB32 (lines17–23). Its
[dataset loader](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/utils/teleop_reset_lib/dataset_wsppt.py)
uses unsorted `os.listdir` order (lines24–36), and the
[reset sampler](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/loco_manip/reset_from_dataset_wsppt.py)
keeps each lane on `env_id % motion_count` (lines40–65). Resetting a lane does
not rotate it through295 motions. With at least32 eligible motions, the default
singleB32 process therefore uses at most32 assigned motions; the exact original
executed population requires its own launch/artifact evidence.

Our all295-motion schedule is an intentional experiment change. The
[sampling plan code](bc_collection/sampling_plan.py), lines50–60 and123–188,
hash-orders the motion inventory independently by role, uses separate process
seeds, selects all training ordinals0–7/8, and preselects validation ordinals
over the same8–9-chunk exposure support. Validation's actual ordinal0–8 counts
are42,39,33,35,41,44,28,39,19. This is same-motion, separately collected data,
not unseen-motion generalization or equal task-stage frequencies. The
[scene binding](deployment/rtx_actor/native_motion_assignment.py), lines110–170,
binds each lane's physical table geometry and reset motion to the same explicit
file. It changes assignments, not the native equations. Cohort/process changes
also change RNG consumption and the frequency of fresh initialization.

### Preserved rollout, reset and label rules

The adapter's `train` checks original collector bytes and calls `super().train()`;
its writer calls the original writer (lines341–391 and468–495). It stops only
after a complete81-vector-step save pass satisfies every lane's preselected
ordinals. It adds no episode reset at81. Original
[`ppo_trainer.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/ppo_trainer.py)
initializes physics once (lines1360–1372), then passes returned observations
between windows (lines1402–1408); model/storage resets at lines711–719 and845
are not physical resets. The initial randomization changes episode length and
elapsed stage0 clocks, not a promise to start each trajectory at task onset.
Reference-frame poses are sampled inside the assigned motion, including random
within-block frame indices; see reset sampler lines224–270.

Original
[`distill_trainer_videogen.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py)
captures controls before action/step (lines203–249), while PPO stores pre-action
observations and teacher action before `env.step` (lines747–775). The writer
appends a row, emits when its episode buffer reaches81, then processes `done`
(lines618–668). It discards incomplete terminal tails, carries incomplete
nonterminal buffers between model windows, and advances sequence IDs across
episode ends. It writes `dones=zeros(81)` even if the81st transition terminates
(line645); this inherited serialization convention is not evidence that every
saved transition is nonterminal. Our sidecars retain the actual terminal flags
and verify within-episode continuity. The11 terminal-at-row81 chunks above are
consistent with this original order, not cross-reset chunks.

Teacher behavior is preserved conditional on the same state: the original
[Homie distillation trainer](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py),
lines314–378, concatenates privileged teacher11 with deterministic Homie15,
including Raven's stage1–3 base-velocity cancellation
`-10 * env._delta_actions[:3]`. This is not raw uncorrected teacher output or the
separate Raven PPO-teacher diagnostic convention. Homie's
[forward](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/modules/homie_modules.py),
lines382–429, returns its distribution mean; the loaded configuration uses
the0.5 walk/stand command-norm threshold. The collection adapter requires
finite26-D behavior actions exactly equal to the teacher labels (lines438–442).
The [simulator config](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/simulator/isaacsim.yaml),
lines15–18, remains200Hz physics with decimation4:50Hz control,81 transitions
per1.62 simulated seconds. Disabling the unused student's optimizer does not
alter the teacher-label function, but is not proof of identical seeded RNG
trajectories to a run that performed learner-side random operations.

### Intentional fixes, inherited behavior and unproven equivalence

The initial robot-refresh correction is an intentional departure from original
Raven, not literal reset parity. Original
[`LeggedRobotBase.reset_all`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/legged_base_task/legged_robot_base.py),
lines662–681, writes robot targets after `reset_envs_idx` leaves pending refresh
flags true; the following post-physics code writes them again (lines700–709).
The [overlay](deployment/rtx_actor/materialize_isaac6.py), lines107–126, wraps
both independent base implementations, including the actual active
`StagedTaskBase -> LeggedRobotBase` hierarchy. The
[completion helper](deployment/rtx_actor/initial_reset_contract.py), lines37–47,
clears those already-applied flags once without stepping or drawing randomness.
Ordinary episode reset bodies remain unchanged. All100 admitted clean cohorts
carry this receipt; the earlier incomplete fix and old exposed data are not
retroactively repaired.

There is **no reward/success/loss selection of complete chunks** in the quota
selection code, but **there is an inherited success-dependent termination
curriculum**. These are distinct facts. The original
[staged task](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/base_task/staged_task_base.py)
computes mean last-goal completion across lanes and updates termination level
(lines241–256 and419–440). The
[task configuration](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/env/walk_stand_place_grasp_turn_homie.yaml),
lines72–78, specifies thresholds.6/.96, degree−.00008 and initial/max level1.
The [task termination](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/loco_manip/walk_stand_place_grasp_turn_homie.py),
lines856–863, multiplies the fast-arm-velocity threshold by this level. The
collector does not disable it; its guard rejects staged-reset/teleop-reset
curricula, not this mechanism. B6/B1 cohorts alter cross-lane success statistics
and restart curriculum history compared with a longB32 scene. The magnitude
of that effect is not established by the saved metadata. Complete81-row-only
retention also favors episodes that survive long enough, in both implementations;
it is not a success-only demonstration filter.

Another inherited sampler edge case remains: dataset loader lines84–93 stores
an upper index equal to the final available block, but reset sampler lines55–64
passes it to exclusive `torch.randint`. Consequently the final block in each
motion's range is omitted from ordinary sampling. The current assignment guard
preserves that behavior; it does not establish uniform coverage of every frame
in the original motion.

The three native conditioning cameras and their calibration are inherited
from the [collection YAML](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_videogen_collect.yaml),
lines24–73:480×832 capture,240×416 controls, `h5_v4` per-view parameters and
textured robot-on-black RGB. Native
[conditioning capture](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/simulator/isaacsim/isaacsim.py),
lines1911–1988, masks robot semantic IDs and performs two render refreshes,
without an extra physics step. A subtle inherited normalization dependency
remains: wrist percentiles are computed over each81-vector-step rollout
**before** within-episode chunk reassembly (collector lines468–520 and546–668).
An emitted chunk can therefore contain rows normalized using different
rollout-window bounds; normalization is not recomputed independently for that
emitted clip. Independent offline generation removes generated-reference
chains, not these within-collection temporal dependencies.

Isaac6 also changes the engine/backend boundary. The overlay and
[compatibility helper](deployment/rtx_actor/roborender_isaac6.py) translate
quaternion conventions, asset/array APIs, scene spawn paths and semantic-ID
rendering. Those are explicit port changes, not proof of identical contacts,
images, reset realizations or teacher success. Source similarity and native
completion receipts support the stated collection/data-flow properties only.
The focused CPU re-audit passed53 tests across
`test_isaac6_bc_production.py`, `test_bc_sampling_plan.py`,
`test_bc_capacity_reset_equivalence.py` and
`test_raven_bc_collection_contract.py`; no GPU/native job was launched by this
audit, and no compute resource was changed or released.

### Offline rendering: verified request contract and remaining source gap

The user explicitly retains generated student inputs for both training and
testing. This audit does not change that image contract. Offline BC uses the
teacher50 renders below; online student training/testing uses its separately
audited Image3F interface. Native RGB and depth remain renderer conditions,
not a fallback student observation.

The supplied original
[three-view launcher](vendor/raven-fast-roborender-reference/scripts/data/rerender_3view.sbatch),
lines19–24, explicitly sets `--max_chain 1`, the canonical prompt override,
left/right/head view order and head primary output. The generic
[offline script](vendor/raven-fast-roborender-reference/scripts/data/offline_render_dataset.py)
defaults to six chunks at line44, but that default is overridden by this
launcher. Current independent rendering is therefore not a new departure from
the supplied three-view BC launch. The historical six-clip recipe is different
and remains explicitly labeled legacy in
[`render_recipes.py`](bc_rendering/render_recipes.py), lines47–60.

| Interface | Supplied original BC client/launcher | Current executed clean BC renderer |
|---|---|---|
| View order | `hand_left,hand_right,head_color`; head primary | Same order; slot aliases `ext1=left,ext2=right,wrist=head` |
| Controls | Per-view uint8 depth and textured robot RGB | Same saved arrays, without renderer-side depth renormalization |
| Prompt | `render_prompt.txt` overrides the collection prompt | Same canonical text for every job |
| Randomness | Seed42 on every render request | Seed42 on every independent clip, not a per-clip seed sweep |
| Clip/reference boundary | One81-row clip per request; client sends no reference image | One81-row clip per call; absent initial/previous reference, no anchor |
| Output | PIL bilinear108×192 uint8 per view; head alias `rgb_gen` | Same sizes, interpolation, per-view keys and head alias |

The original client sends only encoded depth, mask, prompt and seed
([`videogen_client.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/utils/videogen_client.py),
lines92–109). The original offline script concatenates the corresponding view
arrays, supplies seed42, and downsamples the returned frames at lines168–197.
Current [worker](bc_rendering/render_bc_teacher_worker.py), lines184–228,
loads `PAI/Wan2.1-Fun-V1.1-1.3B-Control` diffusion/T5/VAE/CLIP assets in
bfloat16 with the step5294 adapter at LoRA alpha1. Its actual model call at
lines296–349 uses three vertically stacked240×416 views,81 frames,50 steps,
CFG5, empty negative prompt and `tiled=True`.81 already satisfies the model's
4k+1 frame-count rule, so no extra source frame is inserted. Reference state
starts `None` for each chain; max_chain1 prevents subsequent-clip inheritance.
The current [reference unit](../renderer/online_image3f/source/vendor/raven-image3f-joint/diffsynth/pipelines/wan_video.py),
lines945–949, returns no reference conditioning when both reference arguments
are absent. Structural zero tensors at lines890–895 are absent-conditioning
placeholders, not black frames substituted for generated student observations.
The [noise helper](../renderer/online_image3f/source/vendor/raven-image3f-joint/diffsynth/diffusion/base_pipeline.py),
lines172–177, creates a freshly seeded generator per call.

Actual evidence is stronger than intended configuration but narrower than
pixel equivalence. The100 completed plans under
[`runtime/clean-bc-20261020/`](../runtime/clean-bc-20261020/)
contain400 lane plans, three empty, and397 successful execution receipts with
`loaded_paths_verified=true`. They cover exactly2,880 distinct source/output
jobs and2,880 independent chains. Every job has seed42, chain_length1,
`inherits_previous_reference=false`, and the canonical prompt text. All100
plans record50 steps and CFG5. For example, the
[executed plan](../runtime/clean-bc-20261020/stream-render-remaining43/batch-0000/worker/render/plan.json)
and [lane receipt](../runtime/clean-bc-20261020/stream-render-remaining43/batch-0000/worker/render/progress/gpu-00-lane-00.execution.json)
bind the [asset/source identity](../runtime/clean-bc-20261020/render-assets/protocol-bindings.json):

- adapter SHA256 `3f09aa23a26ddb2da32d027dc5e883c992c820854bc07020d6d254c806afb01b`;
- foundation-manifest identity `ac8260e9088a3d7afff8b00aabfafbe28a39f3c06dd40f6b8c63738048c266f2`;
- prompt SHA256 `60f420084002c776cf71502a38b2eb2afddd70eb24c410ee76844b08e8d60d17`;
- renderer-source identity `a3cb495f233940d5b7e367827ae05c1467a849c719b0bb78b7f0aa28a8ef52c9`.

Current worker and orchestrator bytes match the executed frozen
`clean-bc-reset-20261020-v2` source copies. The
[teacher asset manifest](../renderer/manifests/offline_bc_video_teacher_50step_step05294.v1.json),
lines4–16, records checkpoint origin, hash and teacher50 role. However, the
original [server launcher](vendor/raven-fast-roborender-reference/run_raven_server.sh),
lines5–14, invokes external
`/svl/u/ravenh/fast_roborender/roborender-acceleration/runtime/raven_render_server.py`.
That server implementation is absent from the supplied pinned GR00T, reference
and renderer roots inspected here. Its internal defaults, possible reference
handling, exact installed dependencies and actual loaded weight hashes cannot
be certified from the client alone. Current execution uses the later published
[Image3F source snapshot](../renderer/manifests/roborender_image3f_source.hf.v1.json),
revision `dd0a8f20bbce56aa2cf111356b4c8cd310b3a9fb`, plus the attested local
source identity above. In particular its multi-view VAE helper at
`wan_video.py` lines752–783 and854–867 batches controls through `single_encode`
when a view fits one spatial tile, including the81-frame controls. This is
not a demonstrated semantic bug, but numerical identity with an unavailable
original server has not been tested. Earlier parameter-equivalence language
must not be read as proof of identical original model internals or pixels.

Two intentional wrapper differences are visible in code. Original offline
script lines170–175 can duplicate single-view controls into missing view slots
or omit all robot masks if any view lacks one. Current worker lines305–326
requires all three real depth and robot-RGB fields. Original lines187–200
overwrite the source NPZ with its copied fields plus generated images; current
worker lines115–154 publishes a separate immutable output, preserving every
original field and adding render/collection provenance. The first training
and last validation real NPZ pairs were spot-checked: all22 original arrays
were identical in shape, dtype and values; generated views were81×108×192×3
uint8 with an exact head alias and distinct left/right pixels. This is a
two-pair payload check, not a new full-dataset scan. The focused reference,
receipt and render-sampling CPU tests passed42 tests.

Two inherited conditioning/provenance caveats remain. First, the original
collector uses INTER_AREA for480×832-to240×416 control/robot-RGB resizing,
not the generated-output PIL bilinear resize; raw metric depth uses separate
nearest-neighbor resizing. Its head h5_v4 bounds are fixed while wrist
percentiles depend on the81-step vector rollout before episode reassembly,
as explained above. Second, both renderers preserve NPZ `prompt` unchanged
even when the actual rendering prompt is overridden. The
[collection YAML](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_videogen_collect.yaml),
line65, describes a plastic bottle and gray tray; the executed
[render prompt](vendor/raven-fast-roborender-reference/render_prompt.txt)
describes the emerald-green can and black woven basket. The plan/prompt hash,
not NPZ `prompt` alone, establishes what text conditioned generation.

### Entire-episode coverage: what the selected81-row clips actually contain

Neither the original writer nor this collection promises complete task-start
through successful-task-end demonstrations. Native resets sample reference
poses; initial episode clocks are randomized. The writer emits only81-row
within-episode chunks and discards shorter terminal tails. This audit grouped
all2,880 selected provenance sidecars by scoped episode ID and required every
step from index0 through a recorded terminal transition, without gaps, before
calling a span complete. A terminal flag does not by itself mean success.

| Selected population | Original training role | Original validation role |
|---|---:|---:|
| Represented native episodes |1,229|315|
| Episodes whose selected data begins at reset index0 |1,229|178|
| Episodes with a continuous selected span |1,229|314|
| Episodes containing a recorded terminal transition |10|1|
| Complete selected reset-to-terminal spans |10|0|
| Rows belonging to those complete spans |1,701|0|

The ten complete spans occupy21 chunks; three start with randomized native
episode clocks already greater than zero. Validation's one terminal-bearing
episode lacks its prefix. The authorized2,880-chunk training union therefore
still contains only ten complete observed reset-to-terminal spans among1,544
represented native episodes. It does not turn2,880 clips into2,880 episodes or
recover missing rows. Exact full-episode counts for Raven's historical dataset
cannot be calculated without those historical payloads/provenance; its source
code establishes only the81-row emission/drop rule, not complete-episode
coverage.

The100 actual cohort lifetimes account for546,750 executed lane-step rows:
421,119 rows in5,199 emitted chunks,104,237 discarded reset-tail rows, and
21,394 unsealed partial rows at the final quota stops. Selection keeps233,280
rows in2,880 chunks, or42.6667% of those executed rows;187,839 complete emitted
rows are excluded by the predeclared quotas. This is collection accounting,
not a measure of learner passes or success. Rendering preserves the selected
row order, but does not restore discarded tails or turn independently rendered
neighboring clips into one temporally reference-conditioned full episode.

### BC training: preserved objective and differences from original execution

The comparison below distinguishes the original single-view launch from its
later three-view extension. The supplied
[offline launcher](vendor/raven-fast-roborender-reference/scripts/train/offline_bc_train.sbatch),
lines17–23, selects
[`wsdpt_videogen_offline_bc.yaml`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_videogen_offline_bc.yaml).
The retained local policy follows
[`wsdpt_videogen_offline_bc_3view.yaml`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_videogen_offline_bc_3view.yaml),
lines27–53 and80, and its shared
[student architecture config](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay.yaml),
lines164–247. The vendor's reported57.3% result is the single-view step11,000
checkpoint on150 ray-traced episodes, not a three-view acceptance target
([reference README](vendor/raven-fast-roborender-reference/README.md), lines90–102).

| Training property | Original Raven offline execution | Current standalone BC learner |
|---|---|---|
| Student inputs | Three-view extension tiles head/left/right vertically; saved actor state | Same generated views at108×192 each,324×192×3 combined, and113-D actor state |
| Teacher targets | Recomputes teacher/Homie during replay, then overwrites storage with the saved26-D labels; action loss uses first11 | Loads saved `gt_actions[:, :11]` directly; no teacher/Homie inference |
| Image preprocessing | Generated uint8 RGB divided by255 and normalized by environment ImageNet mean/std | Same normalization once; no extra resize, crop or image augmentation in this loader |
| Architecture | Trainable ImageNet ResNet18,128 visual features, SiLU MLP512/256/128,11-D action mean; student RMS disabled | Same configured architecture; strict immutable bootstrap load; student RMS disabled |
| Initialization | Constructs pretrained ResNet and randomly initialized projection/MLP under the original run's RNG history | Reuses pinned pretrained backbone and fixed bootstrap projection/MLP; a new seed does not regenerate those weights |
| Supervised update | One full16×81 forward, mean action L2 loss, coefficient1, one optimizer update | Same supervised population and objective; activation recomputation and sample-only max-pool splitting |
| BatchNorm running buffers |81 replay forwards in training mode, then one full training forward:82 updates per ordinary optimizer update | One full training forward updates buffers once; recomputation restores its temporary buffer changes |
| Chunk order | Sorted `chunk_*.npz` paths with a continuous modulo cursor | Sorted indexed relative paths; one complete pass per epoch, cursor0 at complete-epoch resume |
| Within-batch order | Randomly permutes the16 lane indices before its one minibatch | Keeps sorted lane order; same membership does not imply identical reduction order or RNG consumption |
| Incomplete population | Modulo cursor can wrap inside a batch; arrays are sliced to81 rows | Rejects nondivisibility by16 and non16×81 training tensors; no dropped/padded tail |
| Optimizer and auxiliary branches | LR2e-4, constant scheduler, clip1; optimizer delegated to installed Transformers; object loss coefficient0 and frozen action std | Explicit AdamW, LR2e-4, weight decay0, clip1; no scheduler; object head and std excluded from trainable parameters |
| Budget/selection | Shipped default50,000 update batches; reported BC checkpoint is step11,000; no held-out-MSE selection loop | New `fixed_final`: explicit epoch budget, no held-out/train-eval passes, select final complete epoch; historical BC116 protocol remains historical |
| Checkpoint/resume | Policy, optimizer, scheduler, trainer and environment state; offline cursor initializes at0, without an explicit cursor/RNG snapshot in its custom checkpoint | Complete-epoch policy/AdamW/RNG/counters plus data, order, batching and protocol identity; no implicit historical-checkpoint conversion |

Input/label alignment is preserved, with no timestep shift found. Original
[`distill_trainer_videogen.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py),
lines699–747, builds the generated image and saved observation for the same
row, then explicitly writes saved `gt_actions[t]` after parent inference.
The original first11 loss is in
[`distill_trainer_obj_pred_homie_api.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py),
lines502–516. Current
[`train_bc_h200.py`](bc_training/train_bc_h200.py), functions `BCChunks`,
`prepare_batch` and `backward_raven_bc_mse`, reads those same paired fields.
It omits renderer-only controls, privileged teacher state and `dones` because
this configured policy is nonrecurrent and the supervised objective does not
use them. The original actual loss likewise does not mask terminal rows for
this nonrecurrent policy. Data admission and source episode provenance remain
necessary; omitting `dones` from the learner is not a new cross-episode claim.

The [bootstrap comparison receipt](../runtime/production-audits/20261015-raven-faithful/bootstrap-imagenet-audit.json)
records100/100 retained backbone tensors equal to official ImageNet ResNet18V1.
This re-audit verified the current bootstrap SHA256
`720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a` and cached
reference SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
still match that receipt. Construction with `pretrained=false` followed by
strict checkpoint loading does not discard the pretrained backbone. It also
does not establish identical random projection/head initialization to the
original reported run. Vendor comments about three-view weights necessarily
having incompatible tensor shapes are too broad: the configured
[`modules.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/agents/modules/modules.py),
lines303–358, uses adaptive average pooling and a fixed512→128 projection.
Changing image height changes observation distribution and normalization
statistics, not necessarily parameter shapes. The approved fresh three-view
bootstrap requirement and separate score baseline remain unchanged.

### Confirmed original replay-time BatchNorm difference

Matching the full16×81 optimization forward does **not** reproduce all original
offline BatchNorm updates. Original
[`ppo_trainer.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/ppo_trainer.py)
calls `model.train()` at line1332. The offline branch at collector/trainer
lines420–422 directly calls `_offline_fill_storage`, bypassing the normal
parent rollout's `_train_rollout_mode()`; that normal mode switch would have
called `.eval()` at PPO lines1608–1612. The offline method instead executes81
no-gradient student forwards at lines733–739. `torch.no_grad()` does not turn
off BatchNorm running-buffer updates. Upstream
[`unwrap_model_for_generation` in TRL0.17.0](https://github.com/huggingface/trl/blob/v0.17.0/trl/models/utils.py)
and [TRL0.19.0](https://github.com/huggingface/trl/blob/v0.19.0/trl/models/utils.py#L239-L274)
only unwraps/gathers parameters and does not call `.eval()` either. These are
the two conflicting declared dependency versions, not proof of which library
version was installed for the reported run; see the dependency limits below.

The configured ResNet converts its normalization layers to SyncBatchNorm and
sets momentum `.001` (`modules.py`, lines321–337). Original replay therefore
updates these buffers81 times with16-frame populations before the1296-frame
optimization forward. Current
[`raven_learner_worker.py`](dagger/services/raven_learner_worker.py), functions
`checkpointed_full_window_actions` and `_batch_norm_recompute_context`, performs
one full-population forward and restores running mean/variance/count after
backward recomputation. This correctly avoids a *new* recomputation update,
but is different from the original offline replay's82-update history. Earlier
statements about preserving full-batch statistics apply to the optimization
forward/gradient population, not equivalence of all persisted historical
normalization buffers.

The CPU comparison executes extracted original methods with a reduced
SyncBatchNorm policy: original buffer counter82 versus current1 after one
optimization forward/backward; the full-batch training predictions and
parameter gradients match, while running buffers and evaluation predictions
differ. No real-model effect size, success-rate improvement or causal
explanation of BC116 performance follows from this fixture. Current behavior
was retained during this audit; no historical checkpoint was recalibrated or
relabelled. A scientific decision to change normalization needs a new explicit
training contract and experiment identity.

Teacher normalization is a separate issue. Original parent inference uses the
registered teacher/Homie modules, and wrapper `.train()` can reach those
submodules; an unfrozen teacher RMS can update in training mode
([normalizer](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/utils/running_mean_std.py),
lines106–121). Thus the original comment that recomputed labels must always
equal the dataset is not itself proof. The explicit saved-label overwrite at
line747 is decisive: those recomputed labels do not replace the supervised
dataset target. The CPU test uses a deliberately different teacher sentinel
to verify this overwrite; it does not execute the actual privileged teacher.

### Sampling, optimization, stopping and evidence limits

Original `_next_offline_batch` (lines674–684) cycles through the sorted selected
files indefinitely; its initializer selects files with primary `rgb_gen` plus
saved observations at lines159–173. Current admission instead binds explicit
indexes and all three generated view fields. For an identical ordered
population divisible by16, a complete sorted pass and the modulo cursor have
the same batch membership. The authorized2,880-chunk merged training population
has180 full updates and233,280 supervised frames per epoch. The historical
2,560-chunk training role had160 updates; adding320 separately collected chunks
changes data exposure and batch composition, not just a validation switch.
The original modulo wrap for a hypothetical18-chunk pool is separately tested;
the current divisibility gate intentionally rejects such a pool.

Original PPO lines1423–1429 call `torch.randperm` even with one full minibatch.
Original student action-distribution methods also draw samples that are later
discarded in favor of action means
([vision actor](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/modules/vision_actor_critic_modules_obj_pred.py),
lines244–265 and268–299). The current learner calls the deterministic forward
directly. Frozen std `.001` is not added to current targets or predicted means;
removing these draws changes RNG history, not the MSE definition. Since this
configuration has no active dropout, augmentation or recurrent state, these
differences are not evidence of a different stochastic action objective.
They still prevent a claim of bitwise run/RNG equivalence.

The original zero-weight object-position loss is computed before multiplication
by0 in
[`distill_trainer_obj_pred.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred.py),
lines337–379; current code freezes the head and never optimizes that auxiliary
loss. With finite values and zero weight decay, that zero-weight branch adds
no active action-policy gradient. It may still produce zero-gradient optimizer
entries in the original, so optimizer parameter sets are not literally equal.
Current code explicitly constructs AdamW; original PPO lines399–403 delegate
optimizer creation to Transformers. The checked-in config pins the learning
rate/constant schedule, but does not pin the full original installed
Transformers/PyTorch implementation or resolved optimizer parameter groups.
Do not infer cross-version/kernel numeric identity from those hyperparameters.

The original dependency declarations are not a reproducible runtime lock:
[`pyproject.toml`](../runtime/raven_gr00t_collection_94d7985/pyproject.toml),
lines30–32, requires `trl==0.17.0` but leaves Transformers and Accelerate
unversioned; [`gr00t/rl/requirements.txt`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/requirements.txt),
line23, instead requires `trl==0.19.0`. The supplied
[`setup_viral_env.sh`](vendor/raven-fast-roborender-reference/scripts/setup/setup_viral_env.sh)
requests Python3.11, torch2.7.0 and torchvision0.22.0, then installs the project
with `pip install -e .`; that route selects the pyproject declaration unless
later overridden. No original executed-run package freeze or installed
Transformers/TRL metadata was found in the inspected local reference/vendor
artifacts. The current host's sparse qualified-runtime path is not evidence
of Raven's original installed environment. Both upstream TRL unwrap helpers
were checked directly, so this ambiguity does not negate the observed
offline-mode bypass; exact optimizer/runtime replication remains unproven.

The original
[TRL configuration](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/algo/trl/ppo.yaml)
sets50,000 batches and a constant scheduler (lines3–15). Its custom
[checkpoint callback](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/callbacks/model_save_callback.py)
saves the policy, optimizer, scheduler, trainer and environment state
(lines91–110), without an explicit offline cursor or RNG snapshot. Current
`save_checkpoint` and `restore_fixed_final_checkpoint` additionally bind RNG,
complete-epoch counters, exact data/order and protocol identity. Exact resume
within that contract is stronger than the original custom resume semantics;
it is not a guarantee across unpinned library/hardware changes.

Historical held-out validation, patience and minimum-validation checkpoint
selection were requested extensions, including BC116's later user-directed
continuation and stop. New
[`fixed_final_contract`](bc_training/stopping_gate.py) requires an explicit
positive epoch budget and selects the last complete epoch, with no held-out or
train-evaluation pass and no implicit conversion of historical checkpoints.
Existing validation provenance and the historical selected BC116 stay intact.
Removing validation does not itself mean all former validation chunks enter
training: that requires the separately authorized, hash-bound merged index.
No success-test result chooses the training budget or checkpoint.

The new
[`test_bc_raven_offline_semantics.py`](tests/test_bc_raven_offline_semantics.py)
passed all4 CPU tests: original train-mode entry, paired image/state/label
alignment including saved-label overwrite,82-versus1 BatchNorm updates with
matching full-batch gradients, and within-batch modulo wrap. It AST-extracts
the original methods and rejects changed source bytes using
`distill_trainer_videogen.py` SHA256
`cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b`
and `ppo_trainer.py` SHA256
`f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61`.
Missing local reference files skip explicitly. Tests do not fetch remote code,
construct the native teacher/ResNet/simulator, run on a GPU, certify throughput
or establish policy quality. No training behavior or compute allocation was
changed by this training audit.

## Complete DAgger path re-audit, October25

This read-only comparison covers the new continuous fixed100 source path, not
the stopped historical59-update run. It compares the pinned GR00T commit above
with `coverage_production.py`, `coverage_campaign.py`,
`continuous_production_collection.py`, `production_segment.py`, the resident
Isaac adapter, render/student service and standalone learner. The supplied
[single-view launcher](vendor/raven-fast-roborender-reference/scripts/train/image_dagger_run.sbatch)
uses one process with16 environments (lines26–33). The later
[three-view launcher](vendor/raven-fast-roborender-reference/scripts/train/dagger_train_3view.sbatch)
also defaults to16 (lines41–45,104–115), although the three-view YAML default
is32. A launcher's source default is not proof of every historical executed run.

### Collection, batch formation and optimization

| Component | Pinned Raven | New continuous path | Assessment |
|---|---|---|---|
| Physical lifecycle | One initial reset, returned observations continue between81-step windows; native episode ends reset lanes | One native command/scene per RTX per100-update round; no scheduled reset at81,162 or qualification update4 | Previous forced two-window replacement removed in source; natural episode continuity preserved |
| Model lifecycle | Begin/clear student and teacher rollout buffers once per81-step window | Same per-window lifecycle; separate from scene/episode reset | Matched control lifecycle for this nonrecurrent policy |
| Motion support | Lane index modulo the loader's motion list; reset samples within that assigned motion | Explicit first32 inventory-ordered motions, one fixed motion per lane throughout each round | Fixed-lane rule preserved; exact original motion ordering/population is not reproduced or certified |
| Supervision | Privileged teacher labels every visited state;11-D manipulation label plus15-D Homie tail | Same pinned teacher/Homie function and stage-dependent label transforms; target is first11 dimensions | No behavior-action/teacher-label substitution found |
| Rollout windows |81 vector transitions; windows can cross episode resets; no BC-style terminal-tail discard | Same81-row lane windows, with actual episode/step/outcome ledger | Not restricted to first81 steps of a new task; not a promise every episode completes before update100 |
| Batch membership | Current16×81 rows, one learning epoch/minibatch; no prior-window replay | Current32×81 rows from both actors; one joint learner update, no BC/prior-round replay | Requested scale changes the update population and BatchNorm statistics |
| Batch layout/order | Storage[T,B] is transposed to[B,T]; random permutation of lane indices before full minibatch | Stack lane chunks as[B,T] in sorted chunk-ID order; no random lane permutation | Same environment-major flattening, different reduction/RNG order |
| Action objective | Mean squared error over all rows and first11 action dimensions, coefficient1 | Same uniform full-population mean action MSE; no success or reward weighting | Preserved objective; this is not teacher/student state-MSE |
| Training mode | Rollout `.eval()`, learning `.train()`; one full B×81 optimization forward | H200 inference `.eval()`/inference-mode, learner `.train()` for one full B×81 forward | Online DAgger does not have the offline BC replay82-versus1 BatchNorm difference |
| Optimizer | Configured LR2e-4, clip1, constant scheduler; creation delegated to installed Transformers | Explicit AdamW, LR2e-4, betas(.9,.999), epsilon1e-8, weight decay0, clip1, constant LambdaLR | Configured action update matches; original installed optimizer implementation/groups are not fully pinned |
| Checkpoint barrier | Next rollout uses the updated policy; resume loads saved optimizer/scheduler if present | Both actors wait for immutable candidate; full policy and AdamW moments/counters are carried; next rollout cannot use stale parent | Matched update-before-next-rollout ordering, stronger admission; not a native simulator-state checkpoint |
| Budget/selection | Shipped default50,000 update batches; supplied image launchers use teacher behavior; no held-out-MSE stopping | Four requested mixed-beta rounds,100 accepted updates each, no validation stage/MSE gate; select final100 | Explicit experiment change, not the supplied Raven experiment |

Source anchors: original
[`ppo_trainer.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/ppo_trainer.py)
lines711–775,845–847,1360–1372,1402–1452 and1608–1626;
[`distill_trainer.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer.py)
lines241–280 and340–356;
[`distill_trainer_obj_pred_homie_api.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py)
lines314–378,423–425 and502–516. Current lifecycle/batch barriers are in
[`production_segment.py`](dagger/rtx/code/src/dagger_rtx/production_segment.py),
lines307–374, and
[`resident_isaac_trainer.py`](deployment/rtx_actor/resident_isaac_trainer.py),
lines1088–1119. Current native labels and physical cursor advancement are in
[`native_entrypoints.py`](dagger/rtx/code/src/dagger_rtx/native_entrypoints.py),
lines346–365 and392–418. The learner
[`load_batch`](dagger/services/raven_learner_worker.py) is at lines152–181,
full-population loss at334–392, policy/optimizer construction at705–721 and
training/clip/update at818–848. Sorted request order is explicit in
[`server.py`](dagger/services/server.py), lines1254–1259.

The original ResNet converts to SyncBatchNorm with momentum.001
([modules.py](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/agents/modules/modules.py),
lines321–337). One learner's complete32×81 forward therefore has a different
normalization population from Raven's16×81, even though neither implementation
uses time-sliced gradient accumulation. Current activation recomputation, when
enabled, restores temporary BatchNorm buffer mutations; the new plan explicitly
sets `activation_checkpointing=False` and retains full-policy activations
(`coverage_production.py`, line564). Index-safe max-pool splitting is
sample-independent and does not split normalization reductions. CPU tests
support these tensor/gradient properties, not B32 native memory qualification.

Student RMS is disabled and the policy is nonrecurrent with default rollout
history1. Teacher RMS is loaded and evaluated, not trained by the learner.
The current direct deterministic student forward is equivalent to the original
selected action mean conditional on the same weights/input for this architecture;
it does not execute the original discarded `Normal.sample()` draws
([vision actor](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/modules/vision_actor_critic_modules_obj_pred.py),
lines268–299). Frozen std.001 therefore is not added to behavior actions.
Original zero-weight object loss is computed; current code freezes that head
and omits its zero-weight backward branch. Under finite values and zero decay,
this does not add an active action-gradient term, but literal optimizer groups
and RNG histories differ. Unpinned original packages/kernels remain a numeric
equivalence limit, as documented in the BC audit above.

### Remaining image, sampling and lifecycle deviations

**Generated inference versus learner storage is an actual original-code
divergence, not merely naming.** Original
[`distill_trainer_videogen.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py),
lines223,344 and416–418, clones/rebinds the image observation locally for
`policy_step`. Original PPO lines747–749 subsequently store the caller's
unchanged native observation. The comment claiming both inference and storage
see generated images is contradicted by that data flow. Current
[`render_student.py`](dagger/services/render_student.py), lines555–587,
uses generated `policy_rgb` for prediction; the gateway persists that exact
array at [`render_student_server.py`](dagger/services/render_student_server.py)
lines1641–1642 and assembles it into learner chunks at2006–2014. Reproducing
the original storage behavior would conflict with the user's earlier
generated-only train/test contract. No such reversal was made by this audit.

Original per-episode observation mixture is50% native/50% generated; new
continuous fixed100 requires100% generated Image3F5 policy inputs. Original
three-view online tiling, float conversion, bilinear resize and ImageNet
normalization are retained in `render_student.py`, lines515–575. This matches
the three-view extension, not the reported single-view checkpoint. Native
images may still condition the renderer; that is different from supplying
them directly to the student. Anchors remain fixed for each physical episode.
Current persistent episode IDs also fix the original anchor-refresh edge at
an episode end exactly on row81, where original PPO reinitializes local
`dones` tozero for the next window. This intentional fix is not literal Raven
behavior.

The original inherited `enforce_teacher_rollout=True`,
`ratio_teacher_rollout=.5` forces the first half of lanes onto teacher actions
before the later beta draw (student YAML lines191–194; Homie trainer
lines473–495). If a mixed beta were requested there without disabling that
override, its effective teacher fraction would be `.5 + .5*beta`, not beta.
The supplied image launcher uses teacher behavior already, so that distinction
does not change its teacher-only actions. Our requested75/50/25/0% schedule
does not copy the forced-lane override. Original `torch.rand(B)<beta`
(videogen trainer lines261–270) is replaced by an idempotent per-step hash draw
(gateway lines910–919 or1595–1604). This is per-decision probabilistic-mixture
semantics, not identical seeded RNG or the original executed teacher-only run.

### New finding: environment curriculum is not carried like Raven

The continuous correction fixes artificial resets **inside** a round, but does
not reproduce Raven's environment-state handoff **between** separate runs.
Original [checkpoint saving](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/callbacks/model_save_callback.py),
lines91–109, includes `env_state_dict`; PPO lines1803–1804 restores it.
[`LeggedRobotBase.get_env_state_dict`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/legged_base_task/legged_robot_base.py),
lines1205–1208, adds `log_dict`, including the scalar `termination_level`.
The original base [loader](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/base_task/base_task.py),
lines373–391, restores matching environment attributes. This is curriculum
state, not a complete physical simulator snapshot.

Current learner checkpoint creation (`raven_learner_worker.py`, lines441–462)
saves policy/AdamW/scheduler but does not obtain fresh native curriculum state.
New round setup reconstructs `ResidentTrainingEnvironment` from base config
(`resident_isaac_trainer.py`, lines721–779) and has no corresponding saved
curriculum restore. Initial level is1; original
[`staged_task_base.py`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/base_task/staged_task_base.py),
lines241–256 and419–440, adjusts it from lane-averaged success. The configured
thresholds are.6/.96, degree−.00008, min.1/max1
([task YAML](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/env/walk_stand_place_grasp_turn_homie.yaml),
lines72–78). It affects actual fast-right-arm-velocity termination
([task implementation](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/loco_manip/walk_stand_place_grasp_turn_homie.py),
lines856–863). This is a potentially meaningful remaining mismatch, not merely
telemetry. Its realized effect in an actual run has not been measured.

Separately, two B16 native actors update these lane-population statistics
independently. A single B32 scene would have one cross-lane curriculum statistic;
there is no global curriculum synchronization in this adapter. That is an
algorithmic consequence of distributed scaling, even with one jointB32 learner.
A strict reproduction must explicitly bind/carry the relevant environment
state and decide the intended per-actor/global curriculum rule. Do not silently
restore training curriculum into the independent fixed success-test panel or
claim a checkpoint restores lost Isaac physics/RNG state.

Proposed remediation is **unimplemented**: introduce an opt-in, versioned
training-only curriculum handoff rather than changing historical admissions.
After update100, export only bounded, finite curriculum attributes which the
original `get_env_state_dict` exposes and `load_env_state_dict` would restore;
bind them to the exact round, final checkpoint, source, native GPU/process and
terminal command receipt. The next training round must verify the predecessor
file/hash and ownership, then restore after cold-phase preparation but before
its initial native reset. Missing, mismatched or unsupported fields must fail
closed. Round1 must explicitly bootstrap from configuration because BC116 has
no native curriculum snapshot; do not manufacture one. Neither physical/RNG
state nor evaluation-derived curriculum may enter this handoff. CPU tests
should extract the pinned original getter/loader and prove test isolation,
type/shape validation and unchanged physical buffers before any native trial.
This per-actor design preserves each actor's own history; it does **not** solve
the global B32 curriculum issue. A global rule requires a separately specified
synchronization/population policy and native qualification, not an untested
average of two independently evolving termination levels. No code, schema or
active owner was changed for this proposed remedy.

### Verification and deployment boundary

The focused CPU command covering `test_dagger_raven_continuity.py`,
`test_raven_learner_batching.py`, `test_dagger_generated_observations.py`,
`test_dagger_continuous_actor.py`, `test_dagger_continuous_plan.py` and
`test_continuous_production_collection.py` passed **78 tests in 21.01 seconds**.
The continuity fixture executes SHA-pinned original methods and the current
driver with CPU simulator/model doubles; image provenance tests reproduce the
original clone/rebind mismatch. These tests do not execute actual native
physics, Image3F or a complete real-model learner update.

The one-lifecycle-per-round correction is implemented in current source and
CPU-tested, **not deployed/native-certified by this audit**. Current-source
capacity, prefix learner/W&B/resource receipts and same-command continuation
must still establish admission. Historical frozen releases and stopped-run
receipts are unchanged and cannot prove this new path. The retained fixed100
runtime HANDOFF explicitly records its pending DAgger authorization as
superseded by the later evaluator-scaling task; no historical ACK or old
qualification authorizes replay. No production code, live workload or compute
allocation was changed during this audit; only this evidence record was added.

## Complete success-evaluation re-audit, October25

The user confirmed that generated student inputs remain required. The original
goal predicate is preserved; the complete evaluation **experiment is not
identical**. Do not describe the old fixed295 or new scaled panel as Raven's
unchanged default experiment.

| Component | Pinned Raven | Current implementation / boundary |
|---|---|---|
| Policy | Frozen eval-mode student, deterministic action mean, Homie15-D tail | Same student/Homie control convention; strict checkpoint tensor identity, zero optimizer updates, unchanged-state receipt |
| Images | Native ray-traced policy images in original success evaluation | Generated Image3F5 only, per the user's explicit exception; missing renderer fails before policy action/physics step |
| Closed loop | Observation → policy mean → one native env.step → next observation until done | Same order with rendering inserted before the policy, and exactly-once distributed receipt guards |
| Success | OR-latch `grasp_goal_reached` over the episode, count on done; include failures in denominator | Same metric methods plus independent first-episode accounting; no MSE threshold defines task success |
| Terminal edge | A goal achieved on the same step as automatic reset can be erased before export | Pre-reset terminal-goal snapshot preserves that event; native reset/task predicate remains unchanged |
| Population | Default pooled150 completed episodes; supplied evaluation launcher uses4 lanes; optional first-episode-per-lane mode exists | Fixed295 explicit motion cases, first episode per scored lane; scaled B16 changes grouping and RNG realization, not just throughput |
| Initial clocks | Configured `init_at_random_ep_len=True` randomizes episode and stage0 clocks after reset | Explicit panel sets this False and starts fresh clocks; this is a real initialization-distribution difference beyond the corrected robot/object refresh |
| Initial poses | Original reference-frame sampler with implicit lane-to-motion mapping | Same native sampler but explicit motions, per-cohort seeds and dataset resampling; not identical RNG draws or historical initial states |
| Warm isolation | Original reset_all within the original runtime | Qualified panel reset also clears histories, controls/contact caches and event timers, refreshes kinematics without an extra integrated physics step; distinct warm-cohort isolation logic |
| Completion | Continue until configured completed-episode count | Require every declared first episode;450-step guard fails incomplete rather than converting missing cases into failures |

Source anchors: original
[evaluation loop](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py)
lines564–835; inherited student config
[`init_at_random_ep_len`](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/config/exp/loco_manip/wsdpt_student_for_teacher_v8q8.002_resnet_rgb_delay.yaml)
line191; original
[metric methods](../runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs/loco_manip/walk_stand_grasp.py)
lines972–1145. Current
[cohort setup](deployment/rtx_actor/isaac6_motion_evaluator.py) lines115–172,
[diagnostic loop](deployment/rtx_actor/isaac6_success_diagnostic.py) lines234–490,
[first-episode accounting](deployment/rtx_actor/success_diagnostic_core.py),
[terminal-goal fix](dagger/rtx/code/src/dagger_rtx/terminal_goal_accounting.py),
[warm panel reset](deployment/rtx_actor/isaac6_panel_evaluator.py),
[generated observation contract](deployment/rtx_actor/rendered_success_observations.py)
and [scaled protocol](deployment/rtx_actor/scaled_success_contract.py) carry the
separate controls. The scaled protocol explicitly records
`historical_fixed295_equivalent=False`.

The new
[differential evaluation tests](tests/test_raven_evaluation_equivalence.py)
execute hash-pinned original metric methods on the same CPU transition streams
as current accounting. **9 tests passed**, including1/2/7/16/32 lanes, transient
goals, terminal goals, first-episode failures followed by later successes, and
unfinished successful episodes. They also verify the original deterministic
mean-action selection and execute its random initial-clock assignment to
demonstrate the distribution difference. The metric comparison uses Raven's
supported first-episode mode; it does not claim equivalence to default pooled
episode sampling or numerical Isaac behavior.

A separate focused run of evaluation/reset/render/continuity suites passed
**147 tests,1 skipped**. The skipped native-control resize/API test requires
OpenCV, absent from the CPU test environment. No native renderer/evaluator was
launched. The previous159-test union/BC/fixed-final/storage run also passed;
these results precede any subsequently documented student-lifecycle correction.

Adopting randomized initial clocks, original pooled sampling or training
curriculum in evaluation requires a **new explicit benchmark version**, not
editing an existing295-case receipt or relabeling an old score. The generated
observation exception does not erase these remaining sampling differences.

### Implemented BC correction after the source comparison

New `fixed_final` runs default to versioned `raven_offline_replay_v1` in
[`train_bc_h200.py`](bc_training/train_bc_h200.py). It calls the pinned student
for81 train/no-grad replay steps, including discarded Normal draws, clears
rollout history, permutes lanes and performs the full16×81 learning forward.
The zero-weight auxiliary object loss also creates the original zero-gradient
head optimizer state. Activation recomputation preserves RNG/BatchNorm state.
Historical direct-supervised/BC116 runs keep their old identity; incompatible
resume/finalization is rejected rather than relabeled.

The two-update hash-pinned CPU comparison matched gradients, parameters, BN
buffers/counters, RNG, permutation and AdamW state bit-for-bit with the same
explicit PyTorch optimizer. An actual fixed-final CPU loop/checkpoint/resume
fixture also passed. This does not prove equality to the unknown historical
Transformers optimizer build, omitted teacher/environment RNG consumption,
real ResNet CUDA numerics, or a success-rate improvement.

All2,880 actual rendered files were checked for finite `obs_gt_obj_pos`: every
one contains81×3. The replay loader accepts original wider layouts too and
uses the first3 coordinates. No payload was changed. The standard launcher
now resolves the merged union from the central registry; explicit alternate
pools require a complete coherent four-path override, with legacy defaults
only under explicit `legacy_validation`.

The final combined union, launcher, BC lifecycle/batching/checkpoint/handoff,
evaluation accounting/action-guard and storage test run passed **228 CPU tests**
(11 expected no-accelerator pin-memory warnings). Shell syntax and diff checks
passed. No training/evaluation launch, source-release deployment, checkpoint
replacement, commit or compute-allocation change was performed.

### Additional evaluation-control finding and tested safeguard

Selecting `action_mean` is **not sufficient** to guarantee student-only
evaluation in the supplied original code. The Homie trainer's
`enforce_teacher_rollout` branch (lines473–495) is outside its training-mode
guard and overwrites both actions and means for the configured teacher lanes.
The inherited student YAML defaults to true with ratio.5. The exact saved
configuration used for Raven's published evaluation is unavailable here; this
finding does **not** establish that her reported score was teacher-assisted.

Our supported native launcher already explicitly disables the override with
false/0/0 settings. The new
[`require_student_only_evaluation`](deployment/rtx_actor/success_diagnostic_core.py)
guard requires those settings at the student evaluation entrypoint, before
checkpoint/model/reset/step work. Missing or inherited unsafe flags now fail
closed. Explicit privileged-teacher diagnostics are unchanged.
[The new test](tests/test_student_evaluation_action_guard.py) executes the
hash-pinned original policy-step method in eval mode and reproduces first-half
teacher takeover, verifies the safe student's11-D mean plus15-D Homie tail,
and verifies rejection before native work. A combined control/accounting/phase
regression run passed148 CPU tests. No live evaluator was changed or launched.

The current custom diagnostic loop also does not contain Raven's initial-clock
randomization block. Merely flipping its config flag would do nothing: a new
version must explicitly draw episode and stage clocks after reset and before
the initial-state receipt. This remains unimplemented and requires a distinct
benchmark admission. Pure-student evaluation and generated-only pixels stay
mandatory; historical runtime claims are not revised by this source safeguard.

### Curriculum carryover correction and unified certification release

The future-source native/provider/campaign paths now preserve the restorable
intersection of Raven's `get_env_state_dict()` and `load_env_state_dict()`:
training curriculum scalars, not physics or lane history. The new
`dagger_rtx/training_curriculum.py` exports only an exact completed update100
production boundary (or separately admitted update4 diagnostic boundary), bound
to each actor/source/checkpoint/native command. The next training phase restores
that export after cold preparation and before its original initial reset. Tests
execute the hash-pinned original getter, loader and reset curriculum update and
compare the effective state after the reset, not merely the saved value. Test
curriculum and simulator state are never imported into training.

This fixes missing round-to-round carryover. It does not make two independent
B16 actors identical to one B32 population: curriculum thresholds use each
actor's population. The original strict greater-than.96 threshold requires16/16
forB16 but31/32 forB32. No averaging or hidden global curriculum is introduced.
Generated inputs, fixed explicit test cases/fresh clocks, Isaac6 and distributed
execution remain documented differences; published-score equivalence is not
claimed. The supported fixed-final BC default is now32 chunks (90 updates per
merged2880-chunk epoch), with explicit historical16 overrides still supported.

The pending live certification uses one full frozen source for native and
learner plus versioned diagnostic scopev2; old split-source/v1 results remain
historical evidence. CPU checks and prepared plans alone do not certify the
train→test→train→test cycle. Require actual generated-image32-lane updates,
candidate-specific scaled tests, exact policy/AdamW/RNG/curriculum continuity,
clean phase receipts and unchanged warm native identities through completion.

### User-authorized separate pooled benchmark (live qualification pending)

The user chose “Add a separately versioned Raven-style benchmark”. This
supersedes the earlier statement that initial-clock randomization was not
implemented, but does not change the fixed295 benchmark. The new guarded native
operation explicitly draws episode then stage clocks after corrected reset,
copies the stage clock to `actual_time_in_stage_buf`, and pools every completed
episode through the final vector step, including overshoot. Each RTX remains an
independent B16 replicate with150 completed episodes as its minimum target;
bounded qualification uses32 per actor. Actual original-code differential tests
and source-bound live receipts must support any certification claim.

The new path has CPU differential coverage for Raven's original clock draw
order, repeated-lane metric callbacks and final-step overshoot. Natural episode
reset evidence is captured at `reset_envs_idx`, not each task-state reset.
Every pooled iteration checks the installed resource guard before rendering,
before policy inference and before physics. Its completion envelope binds the
actual command, worker, source, frozen checkpoint, unchanged Kit, render inputs
and resource receipts. These checks do not substitute for the pending live run.

Raven's dataset maps a lane to a motion and samples reference frames within that
motion; repeated completions do not imply coverage of the full motion inventory.
The new benchmark records its actual16-motion support per actor and does not
claim295-motion coverage or a global B32 pool. Generated-only pixels, corrected
initialization/terminal-goal export, explicit motion assignment, cold evaluation
curriculum and pure-student safeguards remain disclosed differences. Historical
fixed295 outcomes are never imported into pooled results.

### Ordered BC functional comparison — October25, not native certification

The new `test/pipeline/test_bc_collection_functional_parity.py` executes the
entire hash-pinned original PPO train loop, original distillation/video rollout,
collection action wrapper, h5_v4 conversion/resize and NPZ writer directly and
through our actual adapter. Identical scripted inputs use3 CPU lanes,4/8 windows
and zero optimizer epochs on both sides. All emitted NPZ bytes/fields/dtypes,
all26 behavior values (including discarded rows), capture-before-step ordering,
initial/randomized clocks, natural resets and tail accounting match. Independent
episode partitions cover terminal rows80/81/82; outputs contain9/21 chunks.
The original configured incidental learner epoch is NOT tested or reproduced by
this zero-epoch comparison. Physics, teacher inference, storage and framework
services are CPU doubles; this is not full-resolution/native/B32 qualification.

The source-reference tests now use the clean pinned runtime rather than mutable
GR00T. The native receipt additionally attests `time_in_stage_buf` and
`actual_time_in_stage_buf`; ingestion requires their identical1D int64 hashes,
matching Raven's initial copy without modifying clocks or physics.
The consolidated BC-focused suite passed168 tests with no skips. Evidence and
remaining ordered gates: `runtime/raven-stage-certification-20261025-kBFQ3q/status.json`.
No live owner, frozen release, dataset, model or allocation was changed.

### Authorized bounded native BC comparison — October26

The separately approved idle-owner replacement ran one source-pinned B2 native
scene for4windows/324vector steps (648lane transitions). It produced five complete
chunks, per-lane[3,2], with one initial reset and native done/timeout continuity.
The original writer replayed the exact captured full-resolution inputs in a
CPU-only container: all five NPZ files, field values/dtypes/order, emitted counts
and carried tails matched exactly, including byte hashes. This does not compare
two independently reproduced Isaac trajectories or the original incidental
learner/RNG work. B32 and long-campaign qualification remain separate.

Peak cgroup27.70GiB under40GiB; minimum available host31.69GiB; zero OOM, limit
and owned-swap events. The strict comparator first exposed its own hard-coded
camera dictionary order. A test-only correction derives the omitted order from
independently bound original YAML/native Hydra config/ported source for the old
capture, and records order directly in future captures. It does not change native
data or relax equality. Both failed comparison attempts remain preserved;23
focused regression tests pass. Evidence:
`runtime/raven-stage-certification-20261025-kBFQ3q/bc-parity-b2-w4-v1/comparison.json`.

The success-stage preparation adds a full pinned original eval/policy-step versus
current diagnostic-loop comparison on matched CPU inputs;76 evaluation tests
pass, including7 new paired cases. First-episode and pooled protocols are tested
separately, with per-episode reward/length/action equality and pooled overshoot
retained. A scripted terminal-reset case explicitly differs (original0/4,
corrected1/4); it is not relabeled equivalence. Physics, renderer and neural
networks remain test doubles. No matched live evaluation or DAgger certification
follows from these CPU results.

### Actual recorded teacher scoring comparison — October26

The test-only `test/deployment/rtx_actor/evaluation_trace_replay.py` executes the
hash-pinned original metric callbacks and first-episode completion/reset block,
feeding actual raw exported goals, natural done flags and recorded cumulative
rewards from all19 completed teacher cohorts. Across7,243vector events and
113,314lane-events, both scorers match every case's outcome, reward and episode
length:255/295successes (86.4407%), preserving all40failures. Every raw result,
phase wrapper, command commit, case membership and aggregate is reconciled.
Evidence: `runtime/raven-stage-certification-20261025-kBFQ3q/evaluation-trace-root-frozen-comparison.json`.

This is matched first-episode scoring of historical real events, not two new
simulator runs or Raven-default pooled evaluation. It does not retest policy,
renderer, reset physics, velocity metrics or resource capacity. Full trace
hashes were freshly observed; the historical receipt sealed only a numeric
prefix, not those complete trace bytes. Do not claim stronger provenance or
independent live equivalence. The first replay rejected unbound copied wrapper
files only after matching all raw events; the test helper now verifies those19
wrappers too instead of ignoring them. No native work or data was replayed.

### Coupled DAgger CPU comparison — October26, deliberately conditional

`test/pipeline/test_dagger_coupled_functional_parity.py` joins pinned original
PPO/Distill collection and Homie/VideoGen policy/loss methods to a comparison
against the actual current window runner, action selection, artifact sealing,
chunk loading and full-population backward method. B4 runs3×81 transitions at
beta.75; natural terminals occur at80/81/82 while another lane remains live past
162 through243. Exact window images, teacher-action labels, done flags and temporal
order match.12 sealed chunks are used; updated tiny-policy weights feed each
subsequent window.69 combined tests pass, including15 new tests and corruption
controls, with zero skips. Required pinned source/dependencies fail if absent.

This is not literal RNG or optimizer-bit identity. The original Bernoulli call
receives an explicit tape of current hash-based selections, while original
lane permutation is retained instead of current sorted ordering. Both receive
generated inputs, deliberately excluding Raven's native-storage relabel issue.
Maximum absolute differences: gradients9.21e-7, policy3.43e-6, optimizer7.67e-8,
loss1.43e-6; near-zero gradients reach8.09% relative difference. Gradient/policy
checks use atol1e-6,rtol2e-4. Moment checks do NOT use that absolute floor: each
side is independently checked against AdamW recurrence, cross-side errors are
bounded by actual clipped-gradient/prior-moment differences and8eps operand
rounding, and zero support/steps must match. Negative tests reject lost moments
even below1e-18. Per-window measurements are in
`runtime/raven-stage-certification-20261025-kBFQ3q/dagger-coupled-root-strict-junit.xml`.

The simulator, renderer, teacher/Homie, network and transport are named doubles.
Original VideoGen._rollout_step/PPO.train, production optimizer/scheduler setup,
NativeRaven, real backend and learner admission/main/checkpoint serialization
are not executed; permutation and optimizer orchestration are test harness
operations. The confirmed current hash selection/RNG and sorted learner order
therefore remain genuine differences, not repaired by this test. Independent
B16 curricula also remain distinct from a global B32 population. No new native
DAgger training or production admission follows from these results.

## October26 DAgger correction: fresh eight-H200 source only

The user's follow-up requests correcting unapproved differences, not relaxing
the explicitly retained generated-input/reset/views/scale/budget exceptions.
Newly generated continuous eight-H200 configs now bind three protocols:

1. `behavior_selection.py`: one native PyTorch vector selection per RTX step,
   before H200 sharding. Consume Raven's discarded B×11 Normal draw and then
   `torch.rand(B)<beta`; beta0/1 consume no Bernoulli draw. Store choices before
   sending requests. Changed requests conflict; lost responses reuse choices;
   a crash before publishing RNG outcome fails closed without re-stepping.
2. `services/learner_order.py`: retain admitted actor/lane order, restore saved
   learner RNG after setup, apply Raven's `torch.randperm(B)` to entire81-step
   lanes, and perform the existing full B32×81 update. Checkpoints/results bind
   the actual permutation and continuing RNG; no legacy DAgger upgrade or
   new-checkpoint downgrade is implicit. **Correction to earlier audit prose:**
   server sorting checked chunk membership; it did not reorder the request
   passed to the learner. Missing permutation, not hash-sorted input, was the bug.
3. `population_curriculum.py`: pair both native reset callbacks, including
   empty local reset sets. Calculate statistics on32 lanes and invoke the actual
   original curriculum methods at their reset boundaries. Preserve unfinished
   episodes, use global completed-task history (not last-stage success), and
   carry shared state between rounds. Scope binds both process instances,
   configuration and source; PIDs may both be1 in distinct containers. Matching
   prefix/terminal evidence is required; no averaging independently evolved
   curriculum scalars after a window is allowed.

Tests are under `test/rtx/test_raven_behavior_selection.py`,
`test/rtx/test_population_curriculum.py`, `test/pipeline/test_raven_learner_order.py`
and the expanded coupled/transport/checkpoint tests. They execute SHA-pinned
original policy, permutation and curriculum bodies, with named CPU physics,
network and transport doubles. The new learner-order branch compares three
updates' model/BN/AdamW tensors and RNG **exactly**, with and without activation
recomputation; the historical branch retains its documented tolerances.
Negative controls cover independent curriculum at31/32 versus15/16 success,
changed batch/order, missing receipts, response loss and uncertain RNG outcome.

**Limits:** no live GPU command was launched for these source changes. The
two-native-process collection RNGs and H200 learner RNG remain separate, not one
bitwise-identical original single-process random stream. Equal random-selection
law and matched local operations do not establish same-seed trajectory identity.
Native reset/motion sampling, renderer/Isaac numerical equivalence, BC incidental
learning/RNG and other outstanding full-stage comparisons remain open. The
earlier conditional CPU result is not retroactively a full Raven reproduction;
neither are these unit/integration results. No retired BC artifact is restored,
no old accepted campaign is reinterpreted, and no compute release is authorized.
# October26 overnight correction — pending live certification

New fixed-final BC sampling v2 removes the non-divisible-pool discrepancy:
sorted full batches wrap modulo the pool and keep their cursor across reporting
epochs. Resume restores the accepted-update cursor, not prefetched requests.
CPU tests compare pinned Raven batch sequences and exact policy/AdamW/BN/RNG
resume for a non-divisible pool. Historical divisibility findings below describe
older releases, not this version. `ceil(N/B)` reporting epochs are explicit;
an epoch MSE is the average of actual exposures, not unique chunks exactly once.

New continuous eight-H200 configs opt into `reset_inventory_protocol` v1. Both
RTX actors keep the complete ordered295 reset-motion inventory; scene lanes
remain0..15 and16..31. Only IDs passed to native `_get_sample_idx` are translated;
physics indices stay local. Native resampling consumes the full inventory.
Receipts must attest inventory hash/count/offset, and reject dropped eligible
motions. The assignment observer now accepts Raven's inclusive final sample
index (its curriculum branch can select it; its randint branch excludes it).
No change to the native sampling formula or physical reset is made.

Evidence/work remaining: `runtime/raven-parity-overnight-20261026-t6oMpN/HANDOFF.md`.
CPU success is not GPU capacity, live continuity or warm-transition certification.
Historical directory order and distributed-vs-single-process RNG/physics
numerical equivalence remain explicit limits; no numerical identity is claimed.

## October26 BC incidental RNG correction — CPU and device verified

New continuous BC plans explicitly bind
`roborender.bc-collection-incidental-rng.v1`. Raven's configured collection
loop runs one full-population learner epoch: despite teacher-only behavior,
that epoch consumes a device `randperm(B)` and a discarded `Normal(B,81,11)`
sample. Omitting those draws changes subsequent reset sampling. The adapter
now consumes the same draws at the original training boundary without fitting
native RGB. A source/architecture/dtype guard rejects unreviewed recurrent,
augmented or stochastic policies. Old requests remain explicitly old protocol;
their results are not retroactively changed or certified.

CPU tests execute the hash-pinned original train/rollout/writer plus its original
Normal-sampling method, with a declared tiny deterministic policy and real
AdamW on the reference side. Four/eight windows, terminals80/81/82 and carried
tails yield identical saved payloads and final RNG state. Separate five-window
tests cover B1/2/16/32/40 and next-reset draws. Evidence:
`runtime/raven-parity-overnight-20261026-t6oMpN/bc-full-loop-rng-differential-v1.xml`
(17 passed); collection regressions before this integration extension138passed.
The separate actual-device probe subsequently passed on each physical RTX:
15 batch/seed combinations x5 windows, exact RNG state and next-reset draws,
using the pinned original Normal method plus the actual policy architecture
guard. `runtime/raven-parity-overnight-20261026-t6oMpN/bc-rng-device-v2/result.json`
binds both device results and unchanged idle native identities. No simulator
command ran in this probe. It does not reproduce discarded student weights,
certify a new native collection trajectory or prove Isaac5/Isaac6 numerical
identity. The fresh v5 owners include this correction; frozen v4 was not patched.
