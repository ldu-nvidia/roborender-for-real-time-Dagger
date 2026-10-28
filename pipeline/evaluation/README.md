# Raven-equivalent evaluation

Current implementation and entry point: [`../success_rate/`](../success_rate/README.md).
This directory retains historical benchmark documentation and compatibility
imports. It is not a second implementation or a new production launch path.

## Separately versioned Raven-style benchmark — October25

The user authorized an additional benchmark, not a replacement or reinterpretation
of fixed295. `deployment/raven_pooled_dispatch.py` plans the distinct native
`evaluate_raven_pooled_rendered` operation; its contract is
`deployment/rtx_actor/raven_pooled_benchmark.py`.

| Property | Fixed295 | Raven-style pooled benchmark |
| --- | --- | --- |
| Scoring population | Each explicit case's first episode | Every completed episode in each independent B16 actor |
| Initial clocks | Fresh clocks | Raven's random episode/stage-clock draws, once after reset |
| Completion target | Exactly295 cases | At least150 episodes per actor; retain final-step overshoot |
| Bounded qualification | Paired capacity ladder/repeat/tail | At least32 episodes per actor; not a full benchmark |
| Motion coverage | All295 fixed cases | Record actual lane motion support; repeated episodes do not add motions |

Both use generated three-view input, a frozen student, corrected initialization
and terminal-goal accounting, no teacher takeover, and no optimizer updates.
The two pooled B16 replicates have separate denominators and a weighted combined
score; they are not one synchronized B32/150-episode pool. Current explicit
motion assignment and cold evaluation curriculum remain declared differences
from Raven's original runtime. Do not pool these scores with fixed295 or claim
published-score reproduction. Implementation and CPU tests alone are not live
qualification; require source/owner/checkpoint/render/reset/resource receipts.

## Scaled evaluator development — October24 user override

The user stopped the running B2 BC116 test to qualify **16 environments per
RTX /32 concurrent cases**, rendered by eight dedicated H200 workers on one
node. Each RTX uses a disjoint four-renderer shard. One all-lane render barrier
precedes each Isaac vector step; no ray-traced policy fallback. The fixed295
motion order is regrouped into18×16+7 under a new reset/panel identity; do not
reuse historical B2 scores or claim unchanged initial-state realizations.
Keep the new panel/config fixed for stage comparisons. Persist complete
cohort receipts and preserve uncertain partial work without replay. CPU tests
are not native capacity certification: require paired full-loop B4/B8/B16,
target repeat, tail7 and actual eight-device headroom/identity evidence. See
the [active contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md).

## Current student input contract — October24

New BC and DAgger success tests use **online Image3F-generated three-view RGB**,
including smoke tests. The corrected fixed295 physical cases/seeds and scoring
are unchanged, but rendered tests have new panel/protocol identities. Rendering
configuration and checkpoint are hash-bound; missing renderer or incomplete
all-lane receipts fail closed with no ray-traced fallback. The active200-case
paired diagnostic and all historical results retain their frozen identities.
Teacher-oracle tests remain privileged-state control tests. Image3F5 is not the
offline BC teacher50 distribution. Read the current
[production contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md); the sections
below describe historical ray-traced benchmarks, not new launch defaults.

**Beta readiness notice (2026-10-12):** the corrected v4 B5 full panel completed
sequentially on GPU1: BC **0/256 (0%)**, three-update candidate **11/256 (4.30%)**.
All 52 paired cohort starts and 104 accounting receipts matched; final metrics
were read back from W&B. These poor absolute scores are not a policy-quality
endorsement. A concurrent GPU1/GPU2 attempt stopped at the host-swap guard
before scoring; that evaluation layout is not qualified. Do not equate this
new panel to the legacy B64 scenarios. Legacy launchers below bind alpha assets. See the
[current qualification report](../deployment/QUALIFICATION_20261012.md) and
[`../deployment/WORKING_REPO.md`](../deployment/WORKING_REPO.md).

The metric is closed-loop task success with native ray-traced observations.
Raven's headline result uses 300 completed episodes. Our checkpoint-validation
panel uses the same metric on the same fixed 256 held-out scenarios for every
candidate iteration; the final test remains separate. MSE is training telemetry only.

Success follows Raven's implementation exactly: OR-latch
`infos["grasp_goal_reached"]` over the episode, append that Boolean once when the
episode completes, and divide successful completed episodes by all completed
episodes. A goal reached earlier in an episode remains a success even if the
eventual termination label is different.

The fixed-panel environment now uses `TerminalGoalAccountingMixin` with
accounting version `roborender.raven.pre_reset_goal.v1`. It snapshots the native
goal immediately after termination checking and exports that snapshot after
auto-reset. This corrects the inherited loss of a goal first reached on a
terminal frame without changing the goal predicate, physics, reset behavior,
or new episode buffers. Historical uncorrected scores must not be silently
pooled with corrected scores. Runtime qualification is recorded separately;
this accounting correction alone does not certify migrated contact physics.

The H200 validation ledger exposes `/v2/evaluations`; use
`../dagger/success_validation.py` to register the fixed panel and atomically
advance the coordinator from its success-rate result. RTX owns Isaac and
deterministic student inference for the ray-traced validation job after the
candidate checkpoint is staged there. H200 owns only the immutable result and
promotion ledger for this path.

`run_raven_raytraced_300_eval_h200.sh` stages the candidate checkpoint beside
the exact `config.yaml` saved by the offline BC baseline, because GR00T loads
the environment configuration from the checkpoint directory. This prevents a
DAgger run's training environment from silently changing the evaluation.

Before staging, `validate_raven_baseline_config.py` rejects a caller-supplied
config unless it is Raven's three-view offline-BC trainer/view/loss contract.
The evaluation log records SHA-256 digests for both checkpoint and config.

Only `num_envs` may change. It affects parallelism, not the fixed total of 300
episodes. The result metric is task success rate.

For the three-view pipeline, register the BC baseline and every candidate on
the same immutable 256-case panel.  Do not compare the historical random
300-episode BC result with a fixed-panel DAgger result.  Export the two complete
evaluation responses and compare them with:

```bash
python pipeline/evaluation/compare_paired_success.py \
  --baseline /path/to/bc-fixed-panel.json \
  --candidate /path/to/candidate-fixed-panel.json \
  --output /path/to/paired-comparison.json
```

The comparison fails closed if the manifest, observation mode, metric, case
count, or scenario identities differ.  It reports the four paired outcome
cells and the exact two-sided McNemar probability, so an apparent percentage
change cannot be mistaken for a paired improvement.

When supplied, accounting version and runtime physics/reset/camera identities
must match too; a one-sided identity is rejected. Unversioned historical
reports are labeled `legacy_unversioned` with incomplete runtime identities,
not certified as equivalent to the corrected evaluator.

The production three-view matrix (BC, privileged teacher oracle, and DAgger;
ray-traced-only and Image3F-only) is specified in
[`THREE_VIEW_FIXED_PANEL.md`](THREE_VIEW_FIXED_PANEL.md). Its v2 assignment
also seals the camera calibration and action-policy identity. Legacy v1
assignments remain accepted only for the original ray-traced student path.

## Versioned beta small-batch panel (full paired execution qualified)

`beta_small_batch_panel.py` defines the user-approved alternative to the old
native B64 panel. It is a **new benchmark**, not a partition of the old cases:

- 52 native cohorts of five environments, one per held-out motion;
- seeds `1103 + 1009 * cohort_index`, fixed before any policy is evaluated;
- all five first episodes are scored in cohorts 0–50; only the preselected
  lane 0 is scored in cohort 51: exactly 256 scored and four unscored episodes;
- at every cohort boundary, reseed Python/NumPy/Torch, call Raven's native
  dataset resampling, clear its resampling counter, then call native reset;
- panel v3 additionally restores cold locomotion commands and zero history,
  refreshes reset kinematics without a physics step, clears previous-episode
  contact-force observations, and recomputes Raven's lift-goal expression from
  the refreshed pelvis pose before constructing the first observation;
- panel v4 additionally clears the native appearance-event cooldown clocks
  and common step counter at each independent cohort, preventing different
  preceding episode lengths from changing reset RNG consumption;
- require byte-identical initial physical/state observation descriptors for
  every BC/candidate cohort, not just equal seed labels;
- no augmentation, optimizer calls or weight changes; deterministic student
  `action_mean` plus the exact Homie tail and pre-reset native goal accounting;
- the current observation protocol is three-view native ray-traced RGB.

`pipeline/deployment/rtx_actor/run_small_batch_success.py` owns the container
lifecycle and host-memory guard. By default it runs both policies sequentially
on GPU1; the experimental `--parallel-arms` option runs BC/GPU1 and
candidate/GPU2 under separate 21 GiB caps, but has not passed the host-memory
guard for evaluation. Do not infer its qualification from the separate
two-actor training test. Each candidate cohort waits for its BC
reference and fails before acting on any mismatched state. GPU0 is forbidden.
`isaac6_panel_evaluator.py` reuses
the tested diagnostic action/goal code and reuses one initialized simulator
per checkpoint. Its two-cohort reset probe cannot emit a complete-panel result.
Only all 52 complete cohorts can produce the 256-case summary.

`validate_comparison` checks final checkpoint/panel/physics/accounting identity,
all ordered case IDs, exact numerators/denominators and inference-only receipts.
The October 12 full run additionally reconciled all initial-state descriptors
and replayed three sampled cohorts through Raven's original metric callbacks.
These checks consume small receipts/logs, not full image/trajectory datasets.

The immutable panel and its SHA bind every evaluation. A later change to
batch size, seeds, reset semantics, observations or motion inventory requires
another version and a new paired baseline. Do not mix these scores with the
legacy B64, random 300-episode or single-view Raven numbers. This protocol does
not itself certify camera geometry or Isaac 5/6 physics equivalence.

## Future production paired B16 route (CPU-tested; not live-certified)

`pipeline/deployment/paired_scaled_evaluation.py` and
`paired_scaled_runtime.py` provide the explicit
`roborender.full295-runtime-bindings.v4` retained-provider route. Historical
runtime v1–v3 and their B2 panels retain their old semantics. A scaled panel
without v4 is rejected, never sent to the GPU0/B2 fallback.

The new runtime adds a `paired_evaluation` `{path, raw_sha256}` reference to a
`roborender.paired-scaled-runtime.v1` record. That record binds `source_sha256`,
the actual paired `native` identities, the retained `renderer_fleet`, an actual
source-bound `cpu_validation` receipt and JUnit, and an existing independent
`resource_sampler` journal; `allocation_release_authorized` must be false.
Native/fleet/CPU/sampler references are absolute physical `{path, raw_sha256}`
bindings. The same admitted source must execute on both RTX owners and the
phase provider. A control-only overlay is not silently treated as that source.

For each newly selected checkpoint, the saved coordinator automatically runs
B4, B8, B16, repeated B16 and the seven-case tail qualification. It reuses
already committed trials only for that exact checkpoint, source, native
lifetimes and rendering dependencies. The actual semantic qualification gate
must pass before the full295 dispatch is created. Qualification results do not
select a checkpoint, enter the test ledger or replace any failed test outcome.
RTX0 is routed exclusively to renderer slots0–3 and RTX1 to slots4–7.

Each trial and final test has both-owner IDLE barriers, immutable single-attempt
submission intents, exact native completion/guard bytes and a whole-interval
joint resource audit. UNKNOWN stops new peer submissions, preserves accepted
work and leaves the sampler and all compute untouched. Re-entry reconciles the
same saved command IDs; it does not restart Isaac or replay incomplete work.
The paired coordinator persists an explicit stop-on-uncertainty policy and
reconciles previously attempted commands before untouched peers; a missing
receipt after a saved POST cannot admit another GPU's new cohort. Historical
coordinators do not silently acquire this new policy.
The final `roborender.paired-scaled-final-result.v1` envelope binds all19 native
cohort commits (18 full16 plus the seven-case tail), actual qualification and
all295 outcomes. The DAgger ledger importer revalidates these records and the
exact final checkpoint/render dependencies before importing success metrics.
The new coordinator alone explicitly admits a bounded512MiB full-result and
whole-test resource aggregate, because the measured B16 receipt sizes can put
all295 above the historical64MiB aggregate cap. Individual command/guard/plan
records retain64MiB bounds; legacy coordinator policy remains byte-identical
and defaults to64MiB. Nothing is truncated or silently dropped.

The independent sampler is a deployment prerequisite, not a boolean readiness
claim or a second watchdog created by the evaluator. Its
`roborender.paired-scaled-resource-journal.v1` binding names `journal`,
`status_file`, `process_receipt`, `sampler_source`, exact source/native/fleet
pins, `interval_seconds: 5`, and false compute-release/artificial-activity flags.
The process receipt records its actual PID, boot, start ticks, command-line
hash/argv, source reference and canonical `sampler_spec_sha256` (the journal
binding excluding `process_receipt`, avoiding a circular hash). Both the
currently running process and fresh samples are checked before new work;
missing/stale/changed telemetry fails closed. An existing sampler must be
explicitly admitted to this new schema; an old diagnostic's boolean or stale
file is insufficient.

This implementation does **not** certify the current frozen live release or
automatically deploy a new source, sampler or fleet. A new source-bound
deployment admission and genuine checkpoint-specific measurements remain
required. The new final-epoch/2880-chunk BC selection uses
`bc_success_dispatch --paired-runtime-bindings <v4.json> --execute`, with
`--selection` and `--output`. Its standalone `paired_bc_success` adapter uses
the same saved paired coordinator, automatic qualification ladder and final295
gate; it requires authenticated online W&B before dispatch and verifies the
actual final metrics by readback before publishing `bc-complete.json`.
The DAgger handoff binds that exact selection, comparison and online receipt.
Legacy pool/source CLI overrides cannot be mixed into this route, and the
legacy B2 command planner still rejects selected.v2. This CPU-tested wiring
does not make the current frozen live deployment production-certified.
Scaled grouping and
reset RNG define a distinct benchmark from historical B2 despite retaining
the original295 motion support; use the same scaled panel for the BC baseline
and all DAgger rounds.

The successful BC32 diagnostic memory policy is not the historical production
learner-capacity admission. In particular, it does not waive that admission's
renderer allowance, minimum headroom or source/mode-specific qualification.

### Separate scaled privileged-teacher baseline

`evaluate_scaled_teacher_panel` is a separately versioned teacher control,
not a visual-student evaluation or a reinterpretation of an existing result.
`scaled_teacher_benchmark.py` binds the unchanged full295 scaled physical
manifest: the same cases, motions, cohort seeds, fresh clocks, corrected initial
robot reset, native goal scoring and first completed episode per lane. The19
cohorts remain18 B16 cohorts plus the real B7 tail, assigned alternately to the
two pinned owners. Completed early cohorts may be committed and reconciled,
but neither a small qualification panel nor a success threshold can replace
the295 actual outcomes. Failures remain in the denominator; no RNG, seed or
benchmark tuning is permitted to obtain a desired score.

Behavior is the pinned privileged PPO teacher plus its pinned Homie policies,
using the original PPO action-mean control convention. A frozen private teacher
copy is created at fresh owner startup before any rollout; its PPO noise-control
changes and rollout state must never alter the shared distillation teacher.
Creation requires the separate launcher `--scaled-teacher` opt-in (alongside
`--scaled-success`), which binds `RESIDENT_SCALED_TEACHER` to the exact teacher
protocol schema. Ordinary scaled student owners do not construct this copy.
Every accepted cohort requires actual before/after teacher/Homie parameter
proof, initial-before-action reset evidence, the installed native resource
guard, unchanged shared-teacher evidence, and zero student forwards, renderer
calls and optimizer steps. The loaded student checkpoint is explicitly a
shadow integrity check, not the behavior checkpoint. Cameras may exist in the
unchanged physical scene, but behavior consumes privileged state and makes no
Image3F requests.

This implementation is not live certification. A fresh source-bound owner
admission and all19 actual completed cohort receipts are required. Historical
B2 teacher results do not predict the new B16 grouping's score. Even a successful
scaled teacher baseline cannot certify Image3F rendering, visual-student
inference, BC or DAgger learning, or published Raven numerical equivalence.
