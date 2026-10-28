# Teacher success evaluation: validated reset correction

## Verified result and scope

The completed, predeclared 295-case privileged-teacher evaluation produced
**262 successes / 295 cases (88.81%)**, retaining all 33 failures. This is not
a BC student score. A second seed schedule was stopped at the user's request;
its partial results are retained but are not a completed benchmark.

The score describes the declared motion corpus, Raven's sampled initial
motion phases, an eight-second horizon, and the original success predicate.
It is not a neutral-start whole-task success guarantee or a reconstruction
of Raven's undocumented historical launch. The same corpus may contain
related motions; a hypothetical IID Wilson 95% interval is 84.71–91.92%,
but the IID assumption is not established and this is not a deployment CI.

## Root cause

Raven's initial `reset_all()` writes robot root/DOF states immediately but
leaves `need_to_refresh_envs` set. After the first physics step, the effective
post-physics callback writes those robot states again. With dataset resets,
the task objects are not rewound with the robot. This introduces a robot/object
discontinuity immediately after the first action.

The explicit evaluator constructs a fresh scene per cohort (147 B2 cohorts
and one true B1 cohort). Thus every scored first episode encounters this
startup path, unlike a long continuous evaluator that initializes once.

The opt-in correction clears only the already-applied pending robot-refresh
flags, after the original initial reset writes and observations and before
the first policy step. It adds no physics step, target write, RNG operation,
or policy update. Ordinary done-driven resets remain unchanged.

In a matched, failure-enriched 12-case A–B–A diagnostic, original behavior
scored 4/12, the correction scored 8/12, and reverting scored 4/12. Initial
arrays and first actions matched. This establishes a causal effect on those
cases, not the population-wide size of the effect: no complete uncorrected
295-case baseline was run. Hidden simulator state equality is not claimed.

## Code and invocation contract

- `pipeline/deployment/rtx_actor/initial_robot_refresh_diagnostic.py` defines
  the one-time correction, strict validation, and per-cohort attestation.
- `isaac6_panel_evaluator.py` applies it at the qualified initial-reset boundary.
- `teacher_action_diagnostic.py` reproduces the original deterministic PPO
  teacher action mean plus the supplied Homie controller, without distillation
  action feedback or student behavior.
- `explicit_motion_panel.py`, `native_motion_assignment.py`, and
  `isaac6_motion_evaluator.py` bind motion bytes, lane order, seeds, first-episode
  accounting, and immutable cohort receipts.
- The resident `evaluate_explicit_panel` operation passes these choices through
  and rejects mismatched identities or an unsafe replay.
- Owned render products are released between scenes while Kit remains resident.

The explicit operation requires the existing pinned panel/checkpoint/output
fields plus these opt-in settings:

```json
{
  "arm": "teacher",
  "teacher_action_mode": "raven_ppo",
  "diagnostic_initial_robot_refresh": "clear_already_applied_once",
  "numeric_trace_steps": 24
}
```

The historical teacher mode above remains restricted to fresh, non-resumed
teacher PPO B1/B2 panels. Its v1 contract and evidence are unchanged. The generic
default remains `raven_original`; production student evaluation explicitly uses
the separate v2 reset contract described below. Training resets are not changed
by either evaluation mode, and this does not establish improved student quality.

The teacher's action-mean parameters, observation RMS and Homie weights are
unchanged. Original PPO may clamp its discarded-sample standard deviation;
that allowed behavior is attested explicitly rather than falsely claiming
all raw checkpoint parameters remain byte-identical.

## Success accounting

For each declared lane, accumulate Raven's native goal flag until the first
native done. Count that first episode exactly once. A goal reached before a
later timeout remains a success under Raven's rule. Native, exported and
original evaluation-metric counts must agree. Do not silently omit failures,
duplicate cases, score infrastructure errors as completed episodes, or stop
a benchmark early because its partial score is favorable.

## Required production evaluation after BC and every DAgger round

`pipeline/deployment/rtx_actor/production_success_contract.py` binds all five
stage-end student evaluations (BC, DAgger rounds 1–4) to one accepted panel:

- Exactly **295 cases**, in the same order, with the same motion assets, cohort
  grouping and simulator-seed schedule: 147 B2 cohorts and one B1 cohort.
- Raw panel SHA256:
  `56f5b2205a840558db477ef75a3ff6e62962af107b6b372ab23320970424ee80`.
- Canonical panel SHA256:
  `627b60f1b82677ebe168c947836b03e7e2eabda24e67e250a32103efc7bab377`.
- Shared panel asset:
  `runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json`.
  The large runtime evidence remains outside Git; do not regenerate a same-size
  substitute or reshuffle its seeds between stages.
- Each stage evaluates its selected student checkpoint with deterministic
  11-dimensional action mean plus the unchanged 15-dimensional Homie action.
  Teacher takeover probability is zero, regardless of the training-round beta;
  there is no optimizer step during evaluation.
- All stages use the same three ray-traced policy views, eight-second episode
  horizon, 50 Hz control and 200 Hz physics. Initialized native timing is checked
  before policy stepping. The original success predicate and first-done/ever-goal
  accounting above remain unchanged.

Production callers pass both settings explicitly:

```json
{
  "teacher_action_mode": "distill",
  "diagnostic_initial_robot_refresh": "student_clear_already_applied_once_v1"
}
```

Live activation requires a fresh frozen native source release and worker identity
containing this student mode and `production_success_contract.py`. Do not apply
it through a control-only overlay or assume an already-running old evaluator
has loaded the new code. Protocol-qualified command/output identities separate
the corrected evaluation from previously completed original-reset commands.
Use the corrected BC dispatcher's returned result binding when preparing the
BC-to-DAgger queue; an old queue path is not silently redirected or overwritten.
Historical campaign states cannot certify already-completed rounds using old
results. Preserve them as historical evidence and prepare a corrected campaign
identity instead of relabeling their results.

Here `distill` is the existing student-evaluator action-mode label; it does not
enable learning or teacher behavior. The new mode invokes the same one-time
pending-refresh correction as the teacher experiment, but emits distinct v2
student control/attestation records. It requires fresh student B1/B2 scenes;
unsafe partial native resume is rejected. Ordinary done-driven resets, initial
observations, targets, policy weights, RNG and first policy actions are unchanged
by the flag-clear operation.

BC dispatch/planning and the production DAgger resident/campaign paths require
this protocol. Result import, online metric logging and round advance reject
old-reset results, another panel, incomplete case coverage, teacher-arm results,
or missing per-cohort correction evidence. These checks also apply when a
controller resumes at logging or round-advance boundaries. Small historical
smoke configurations are not silently promoted to the full production benchmark.

The fixed panel and reseeding bind the same cases/seeds/protocol, not a claim of
bitwise deterministic PhysX or RGB across checkpoints. Each native cohort saves
actual reset indices and initial-state fingerprints for comparison; the aggregate
production gate does not yet compare those fingerprints across all stage runs.

The protocol identity is logged alongside checkpoint and panel identity in
W&B. Success is reported as `successes / 295`; an interrupted panel is incomplete,
not a new denominator. Held-out MSE selects the checkpoint and stops training;
the fixed success panel is a stage-end comparison, not an early-stopping input.
Repeatedly using this panel to choose research changes would make it a development
benchmark, requiring a separate untouched test for a final generalization claim.

Validation status: the reset effect has native teacher A–B–A and complete295
evidence above. Student reset semantics, all five routing paths and rejection of
historical results have CPU regression coverage. A new complete corrected
**student** evaluation has not been run by this integration change; 88.81% is
still the teacher result, not a student baseline.

## Evidence pointers (large artifacts stay outside Git)

Shared-storage evidence root:
`runtime/experiments/20261016-prod-1/bc/teacher-native-ablation-20261017/`.

- Full report:
  `teacher-reset-paired-20261018-v3/confirmation295/final-accounting-report-v1.json`
  SHA256 `963852d316082a7b841ef62aac67f7acd55543e4ca6e539f8f71d87d64fed763`.
- Independent reconciliation:
  `teacher-reset-paired-20261018-v3/confirmation295/independent-final-audit-v1.json`
  SHA256 `c4fe2688adfa54d94b4ffb0bb3aed5728e2b24e276cb4a1948f4828e2d0bdc72`.
- Paired intervention:
  `teacher-reset-paired-20261018-v3/paired-analysis-v1.json`
  SHA256 `0b27d08849ba203a8dcc0db1f8bc2cc7e16b65cf9a521eff67239720fc2d29b5`.
- Unchanged original-panel bytes:
  SHA256 `56f5b2205a840558db477ef75a3ff6e62962af107b6b372ab23320970424ee80`.
- Teacher checkpoint 044500:
  SHA256 `943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38`.
- Raven source commit: `94d79851d9c471b1210be81c75116d9bea835a4d`.

Full cohort receipts, actual reset samples, resource guards, numeric trace
spot-checks, and failed experiments remain in shared runtime storage. This
source commit does not embed checkpoints, datasets, credentials, or run logs.

No completion or workload stop authorizes releasing Cosmos H200 compute;
resource-specific user authorization is required.
