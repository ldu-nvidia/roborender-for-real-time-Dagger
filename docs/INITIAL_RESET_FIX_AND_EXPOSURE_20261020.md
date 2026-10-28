# Initial reset correction and historical exposure

## Native qualification correction (October20, 2026)

The first guarded native BC retry stopped before its first physics step:
`missing or inconsistent corrected initial reset receipt`. Commit `9178f7a`
patched `LocoManipBase`, but the active task resolves `reset_all` through
`WalkStandPlaceGraspTurnHomie -> WalkStandGrasp -> WalkTableStand ->
StagedTaskBase -> LeggedRobotBase`. Consequently that commit did **not** cover
the active collection reset. Its CPU tests were insufficient for that claim.

The follow-up overlay pins and wraps both independent base implementations,
preserving each original body. New tests reconstruct the native class hierarchy
and execute both reset bodies;148 focused tests pass. Source release
`clean-bc-reset-20261020-v2` has source digest
`983441add76361994d1f574aae5231e6a77380e9cffd7acede2bb9f675d771c9`.
Native BC B6 requalification passed:243 vector steps, three full save cycles,
at least two complete emissions per lane, peak32,534,560,768 bytes under a40GiB
cap, no owned swap/OOM and a stable repeated-buffer plateau. Receipt:
`bc-session-leo-clean-reset-20261020-guard-v2-b6/capacity-receipt.json`, SHA256
`921e612fd7b878f2b1c2b458ffece1e4decfc12a3c1765f52fc4290e15782c6b`.
This qualifies the bounded BC path/width, not all DAgger or warm phase switches.
The earlier262/295 teacher result used the separate explicit
post-reset clear and is not evidence that `9178f7a` executed correctly.

## What is fixed

Raven `LocoManipBase.reset_all()` prepares robot/object targets and writes them
immediately, but leaves `need_to_refresh_envs` true. The next post-physics refresh
can write the robot targets again without resetting the objects. This is an
environment reset-lifecycle defect, not evidence of a general Isaac engine bug.

Updated Isaac6 overlays patch both **hash-pinned base `reset_all()` methods only**. The original
body is unchanged between `begin_initial_reset()` and `complete_initial_reset()`.
After all writes, observations and history callbacks, the latter consumes the
already-applied flags once. There are no extra physics steps, RNG draws, action
changes, target changes, or success-criterion changes. Ordinary `reset_envs_idx`,
done/task-stage resets, and continuous 81-step window boundaries are unchanged.

`initial_reset_contract.py` records `roborender.initial-reset-completion.v1` with
a fresh generation for each explicit reset. Failed/incomplete resets invalidate
old evidence and reject replay. BC production/qualification and DAgger native
initialization require completed evidence before stepping. BC cohort metadata
records the receipt; its execution source identity includes the helper. The
derived overlay manifest binds both the patched source and helper hash.

Warm training/held-out/testing scene resets use that same shared correction.
Success testing consumes its fresh receipt instead of clearing twice, preserving
the existing explicit success protocol and byte-equivalent success attestation.
An old overlay mixed with new entrypoints fails closed. Historical Raven/Isaac5
reference sources are retained, not silently edited or certified by this patch.

## Historical exposure: start of BC through end of DAgger

| Stage | Direct reset defect? | Downstream effect |
|---|---|---|
| BC initial scene/cohort reset | Yes, in audited collection source | First physical transition can rewind robot without objects |
| BC teacher rollout and saved conditions/actions | Exposed | First episode's trajectory/labels/images can inherit the altered state |
| BC offline photorealistic rendering | No reset code | Renders the saved conditions; cannot repair the trajectory |
| BC optimizer and held-out MSE | No live Isaac reset | Fit/score the already-collected inputs and labels |
| Corrected BC 295-case success test | This defect was already corrected | Still measures a policy trained on exposed data |
| DAgger held-out collection, each beta | Yes, audited old startup path | Frozen validation population can inherit startup effects |
| DAgger online training, each beta | Yes, at explicit cohort/phase initialization | Both teacher- and student-driven trajectories can inherit the effect |
| DAgger Image3F, student inference, selection and transfer | No direct reset | Consume potentially altered observations; do not repair physics |
| DAgger optimizer and held-out MSE | No direct reset | Optimize/select against the resulting trajectory populations |
| Corrected stage-end DAgger success test | Corrected route required | Tests the selected policy; not a way to repair earlier training |
| Handoff to the next round | No direct reset in checkpoint transfer | Weights/AdamW can carry indirect effects forward |

The defect is **not an 81-frame-window reset** and is not shown to occur on every
ordinary episode reset. Its initial disturbance may propagate through a first
episode; subsequent independent resets need not retain it. Frequency, numerical
severity and contribution to success-rate degradation are different questions.

## Concrete data evidence

The epoch100 BC checkpoint binds 2,560 training and 320 validation chunks. The
audit checked all 880 fresh selected-chunk provenance records and all 100 fresh
cohort-manifest receipt links:

- 560 fresh training chunks: old initial-reset path; **37 contain vector step0**,
  and **57 originate from the first task reset**, including later windows.
- 320 fresh validation chunks: same path; **15 contain vector step0**, and
  **26 originate from the first task reset**.
- 2,000 historical training chunks: insufficient per-chunk reset provenance to
  certify freedom from the defect or count exact exposure. Do not invent it.
- Two bounded raw-to-rendered checks retained actor observations and teacher
  actions exactly; rendering did not repair the source trajectories.
- Completed epoch100 BC success test: **148 valid corrected-reset attestations
  covering295 cases**, score81/295. That score did not use the obsolete reset.

These are exposure counts, **not counts of corrupted frames** or a causal
explanation for the entire MSE/success gap. The evidence is retained at
`runtime/overnight-bc100-20261020/audit-20261020/reset_audit_results_v2.json`,
SHA256 `789826c9f99017da78ab8e236c92425b92a57ecf206b946a2c68cadb5a4634a3`.
The original pinned environment commit is
`94d79851d9c471b1210be81c75116d9bea835a4d`.

## Verification and activation limits

CPU regression tests execute the pinned native reset and post-physics AST with
recording simulator stubs. They compare old/new call order, demonstrate removal
of duplicate initial writes, preserve ordinary done writes, and cover B1/2/6/16/
28/32/64, history on/off, failed resets, duplicate completion, stale receipts,
repeated warm cohorts and teacher/BC/DAgger success-receipt compatibility.
An AST comparison requires the entire base file to be unchanged apart from the
three inserted reset-boundary statements. Materialization verifies source hashes.

October20 verification: **1,106 tests passed** across RTX deployment, DAgger RTX,
BC execution/transfer/handoff/controller and BC production tests (91.54s).
A separately recorded focused suite passed **150 tests** (8.13s); Python
compilation and `git diff --check` passed. The focused JUnit evidence is
`runtime/overnight-bc100-20261020/audit-20261020/initial-reset-fix-regression.xml`,
SHA256 `59caeb5244ad0d1392b32a8e3388ca35a8a0f484c0998146a3ebf9c48e6eab88`.

These tests are **not numeric PhysX or GPU qualification**. Before using the new
version in production: freeze a new release/overlay, qualify actual native BC
collection and DAgger training→heldout/test→training boundaries, and verify
first-step robot/object continuity and exactly-once stepping. Do not patch a live
owner, resume a lost simulator, or relabel old data/checkpoints as corrected.
Old data must be retained as historical; recollection/exclusion needs an explicit
versioned experiment decision. Rerendering alone cannot fix dynamics.

This change does not stop any existing frozen run, release compute, delete data,
or claim that all other contact/physics initialization issues have been ruled out.
