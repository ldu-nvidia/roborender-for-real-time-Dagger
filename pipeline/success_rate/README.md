# Success-rate pipeline

Canonical Leo source: `/data1/workspaces/roborender/source/pipeline/success_rate`.

## Current collection-bound student phase

October28 mixed smoke: the user subsequently selected **50/50 teacher/student
actions, Image3F5 and two completed episodes per lane**,128 scored outcomes on
B32+B32. `mixed_episodes.py:MixedEpisodeEvaluator.evaluate` implements this
distinct diagnostic. All lanes render and run the frozen bootstrap student
before a seeded per-lane/per-step branch selection; only the11-D manipulation
action is mixed, retaining the same15-D Homie tail. The transition trace records
teacher/student/selected actions, masks and natural completions. It is not a
simulator-state checkpoint and cannot authorize physical replay.

`STUDENT_TEST/evaluate_collection_mixed` is separate from pure-student routes,
which still reject mixed arms. A single bound admission is consumed on acceptance;
failed native work cannot be resubmitted under a new command ID. After the test,
the mode backend returns to the same Kit in IDLE. Bootstrap loading selects only
policy_state_dict and never restores saved curriculum/optimizer/RNG metadata.
The paired smoke is launched under `mixed32-image3f-20261028-v3` with pinned
releasev5. Earlier v2 failed during environment initialization: a nested Hydra
override replaced inherited randomization settings. The fix overrides the
global block and preflight checks the complete resolved inheritance in the
native image. No evaluation command was accepted in that failed startup.
CPU tests and metadata/configuration gates passed; live128-outcome
completion and rendering qualification are still pending. The pure-student
three-repeat plan below remains a separate, unlaunched future protocol.

The canonical reference is resolved as `bc_initial_conditions_32` from
`configs/storage_paths.json`. On Leo it is:
`/data1/workspaces/roborender/simulation_assets/bc_initial_conditions/bc-production10000-20261027-074500/manifest.json`.
It preserves the original BC request bytes,32 ordered lane assignments, full295
reset inventory, seed and hashes. It describes motion/reset support, not saved
historical robot/object/RNG snapshots. Both RTX actors reuse these same32 motion
conditions with fresh draws:64 concurrent lanes, not64 different conditions.

`contracts.py:collection_student_plan` defines the requested initial-student
diagnostic. The named renderer is explicit, never inferred from checkpoint stage.
Image3F5 is the causal online path; offline teacher50 requires a separate causal
adapter. Native RGB/depth may condition either renderer but never feed the student
directly. The CPU plan is `ready:false` until native integration and genuine
paired32-lane rendering/checkpoint/owner/resource admission are completed.

```text
require both owners IDLE; install frozen student + bound renderer/motion support
clear phase histories; initialize32 lanes per actor with corrected fresh reset
for each actor concurrently:
    completed[lane] = 0; success_latch[lane] = false
    while any completed[lane] < 3:
        controls = capture current synchronized three-view conditions for32 lanes
        generated = render all32 lanes using the pinned renderer
        require every response matches this actor/lane/episode/step
        action11 = frozen_student(normalize(generated), current actor observations)
        action26 = concatenate(action11, Homie15)
        observations, done, pre_reset_goal = Isaac.step(action26)  # one B32 step
        latch goals and score only each lane's first3 natural completions
        clear terminated lanes' histories/anchors; never reset unfinished lanes
    close test render sessions; synchronize; pause physics; keep Kit alive
require96 scored outcomes per actor; total successes /192 is the final rate
```

This pseudocode is the required behavior, not evidence the new student adapter
has run. Equal-repeat accounting is shared with the measured teacher-mode test
(87/96 per RTX,174/192 combined). That test does not certify rendered student
capacity. Current warm teacher-only owners cannot accept STUDENT_TEST.

CPU entrypoints: `python -m pipeline.success_rate preserve-bc-conditions
--request REQUEST --output MANIFEST`, and `plan-collection-student --reference
MANIFEST --checkpoint INITIAL_STUDENT --renderer online_image3f5 --seeds SEED0
SEED1 --output PLAN`. Neither launches work or changes a running owner.

## Layout

Related modules are consolidated by responsibility. Historical receipt schemas
remain distinct; combining source files does not combine benchmark populations.

| File | Responsibility |
| --- | --- |
| `__main__.py` | CPU-only CLI: plan, validate and aggregate |
| `__init__.py` | Lazy compatibility imports; no second implementation or native startup |
| `contracts.py` | Versioned motion/checkpoint/renderer contracts, benchmark definitions and result validation |
| `dispatch.py` | Teacher/student selection, plans, exactly-once coordination and policy-specific transport selection |
| `runtime.py` | Paired owner I/O, phase barriers, resource journals and execution-evidence checks |
| `resources.py` | Memory/owner telemetry and measured capacity qualification gates |
| `episode_metrics.py` | Goal latching, completed-episode accounting, timing and paired-result comparison |
| `evaluation_loop.py` | Frozen-policy vectorized Isaac evaluation loop |
| `policy_actions.py` | Teacher or student manipulation actions plus Homie |
| `student_observations.py` | Online rendered three-view student inputs; no native-RGB fallback |
| `teacher_episodes.py` | New collection-bound teacher-only, equal-repeat evaluator and CPU accounting |
| `mixed_episodes.py` | Explicit online mixed-action smoke; shares reset/accounting, renderer and phase-owner helpers |

Old module names resolve through a single lazy compatibility table in
`__init__.py`, not separate files or copied algorithms. The merged functions use
explicit prefixes where different protocols previously used the same name.
Existing schema strings, thresholds, denominators and safety gates are unchanged.
All unit/regression tests remain under repository-root `test/`.

## Requested benchmark

Use the exact ordered 32 motion assignments recorded by BC production collection,
retaining its full reset inventory and randomization rules. Draw fresh starts
and evaluate new closed-loop episodes; never replay saved chunks or substitute
the historical fixed295 test panel.

The newly requested teacher smoke uses exactly two naturally completed episodes
per lane: 32 × 2 = 64 scored outcomes. Faster lanes may continue stepping while
the slow lanes finish, but their third and later episodes are not scored.
The safety ceiling cannot turn incomplete episodes into outcomes. This equal-
repeat stopping rule is explicit and differs from Raven's pooled-count stop.

`teacher_episodes.py` is under implementation/qualification: CPU accounting
tests are not a live B32 capacity certificate. Its launcher integration and live
run must pass before reporting this smoke started or certified. Historical
fixed295/B16 admissions do not authorize it.

## Core vector step

The existing shared loop is
`evaluation_loop.py:SuccessDiagnosticTrainer.evaluate()`.
Its inherited `train()` hook calls `evaluate()`; no optimizer update occurs.

1. Read the observations for all admitted lanes.
2. Teacher: use privileged state directly, without student rendering or inference.
   Student: render the current Isaac states, then infer the frozen student.
3. Append Homie's 15 lower-body actions to the 11 manipulation actions.
4. Call `env.step(actions)` once with the whole batch.
5. Latch native task success until each lane naturally terminates. Record
   success/failure, clear that lane's metric history and let native reset begin
   its next episode; other lanes continue.
6. Stop according to the explicitly selected benchmark's completion rule.
   Success rate = successful scored completions / all scored completions.

Lane-wise accounting does not step physics one lane at a time. There are no
81-frame training chunks or optimizer updates in success testing. Success is
the native task predicate—not action MSE and not a teacher judgment.

## Runtime and safety

Keep Isaac Kit warm and physics-idle between compatible admitted phases.
Replacing source requires a fresh pinned lifecycle; never patch a frozen owner
or replay uncertain native work. Student rendering failure stops before stepping;
there is no ray-traced policy-input fallback.

Both RTX cards share Leo's CPU RAM. An idle GPU is not memory admission for a
second Isaac scene. Protect the active BC collector's save headroom and verify
actual host/cgroup/device measurements before launching.

Retain all compute. Stopping a workload is not allocation-release permission.
