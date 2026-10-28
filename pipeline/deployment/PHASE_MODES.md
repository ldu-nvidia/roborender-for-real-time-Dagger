# Explicit Isaac phase interface

## Latest measured state — October 28

Latest addition: `STUDENT_TEST/evaluate_collection_mixed` has an explicit mixed
arm, separate from pure-student gates. User-approved replacement of the two
verified-idle teacher owners is complete; new B32+B32 runtimes are starting in
`mixed32-image3f-20261028-v3` using pinned releasev5. The v2 startup failed before
command admission on a lost Hydra domain-randomization interpolation; v5 fixes
the override and verifies complete resolved inheritance before startup.
Scope is two full episodes
per lane (128 total), Image3F5, frozen bootstrap and50/50 selection. Config/CPU
checks passed; live generated observation round trips and warm return remain
pending. No all-phase or pure-student qualification is implied.

The following measured teacher result is historical and preserved:

The separately pinned `teacher32-mode-three-20261028-v1` runtime passed actual
IDLE→TEACHER_TEST→IDLE on both32-lane RTX actors:87/96 each,174/192 combined.
Both Kit/PID identities stayed fixed; duplicate command delivery returned the
saved receipt without replay. These owners are now warm/IDLE, with only
IDLE/TEACHER_TEST capabilities and their one bounded test admission consumed.

The next requested STUDENT_TEST uses the canonical `bc_initial_conditions_32`
artifact, the initial untrained student and3 complete episodes per lane.
`success_rate/contracts.py:collection_student_plan` defines this explicit B32x2
protocol; `ready:false` means it is a CPU plan, not a live launch admission.
The teacher owner cannot execute it or be hotpatched. A collection-bound student
adapter, genuine renderer binding and fresh source/owner admission remain
required. Online Image3F is implemented; the offline50-step BC renderer needs a
separate causal adapter before it can be selected for closed-loop evaluation.

The earlier implementation record below retains its scope and limitations.

October 28, 2026: source implementation and CPU checks, **not a native
qualification or a completed all-phase warm-runtime migration**. Existing
frozen owners are unchanged. The retained collection-bound teacher smoke is
one-shot, not a server that can accept these commands.

## Responsibilities and current integration

| Mode | Native responsibility | Current selector integration |
|---|---|---|
| `IDLE` | Pause physics, retain Kit and compute; no synthetic activity | Implemented |
| `BC_COLLECT` | Existing standalone teacher/Homie collection, 81-row writer, target/drain, provenance and resource guards | Reserved and rejected until a real resident BC adapter is installed |
| `TEACHER_TEST` | Privileged policy, natural terminal success accounting, no external renderer or optimizer | Existing scaled teacher benchmark route |
| `STUDENT_TEST` | Generated Image3F observations, frozen BC/DAgger policy, native stepping, no optimizer | Existing scaled and Raven-pooled rendered benchmark routes |
| `DAGGER_TRAIN` | Continuous episodes, teacher labels, online generated student inputs, H200 learner, explicit curriculum handoff | Existing continuous production and bounded diagnostic routes |

`BC_RENDER` and `BC_TRAIN` are H200 operations. They are deliberately not Isaac
commands; leave RTX in `IDLE` while those stages execute. A mode name does not
choose a benchmark or prove capacity. In particular it must not silently choose
fixed295 instead of the requested collection-bound repeated episodes. The new
collection-bound teacher/student adapters are not wired into this selector yet.

## Source boundaries

- `pipeline/dagger/rtx/code/src/dagger_rtx/phase_modes.py`: CPU-only exact mode/
  operation schema, immutable parameter normalization and completion checks.
- `pipeline/dagger/rtx/code/src/dagger_rtx/resident_service.py`: existing
  authenticated command queue, generation IDs, idempotency, main-thread native
  execution and durable failures. `mode` is separate from command `phase`.
- `pipeline/deployment/rtx_actor/resident_isaac_trainer.py`:
  `ResidentRavenBackend._run_phase` delegates to the existing admitted native
  stage, checks completion, synchronizes CUDA, pauses physics and verifies that
  Kit has not changed. Stage-owned preparation/reset/drain/checkpoint logic is
  not replaced with a generic reset.
- `test/rtx/test_phase_modes.py` and
  `test/deployment/rtx_actor/test_resident_phase_transitions.py`: CPU contract,
  scheduling, contamination and failure-boundary checks. No test code is
  imported into production.

## Command shape

The outer command is unchanged: `roborender.isaac-resident.v1`, exact
`worker_instance_id`, unique `command_id`, `operation: run_phase`, and payload:

```json
{
  "schema": "roborender.isaac-phase.v1",
  "mode": "STUDENT_TEST",
  "operation": "evaluate_raven_pooled_rendered",
  "parameters": {"arm": "bc"}
}
```

This example is intentionally not launchable: `parameters` must also contain
the existing operation's complete admitted artifacts, owner bindings, checkpoint
hashes and renderer configuration. The selector neither manufactures admissions
nor weakens their validators. Normalization preserves the outer command and
worker IDs for the native identity check. There is no default benchmark.

## Lifecycle and invariants

1. Resolve the exact mode/operation pair and snapshot JSON parameters. Validate
   native configuration, artifact identities and owner-bound admission before
   acceptance. Reject unsupported BC integration before touching Isaac.
2. Accept only while the owner is idle. A duplicate ID with identical bytes
   returns the previous acceptance; it does not execute again. `IDLE` is not an
   interrupt or an implicit cancel.
3. On the native main thread, revalidate and delegate preparation and execution
   to the existing stage. Continuous DAgger requires its explicit curriculum
   handoff; old two-window commands cannot enter through `DAGGER_TRAIN`.
4. Require actual complete stage evidence. Collection and tests require exactly
   zero optimizer steps and matching receipt purpose. A partial command cannot
   be converted to success by this wrapper.
5. Synchronize CUDA, pause physics, verify paused state and unchanged Kit, then
   return to idle. Resetting an episode or closing Kit is not a finalizer.
6. On failure, request a pause without reset/replay; the owner persists
   `CRASH_UNCERTAIN`, retains the failed mode and rejects later work. A pause
   error does not erase the original fault.

The stage remains responsible for render-session/anchor expiry, policy and Homie
histories, action queues, dataset membership, RNG and curriculum boundaries.
The wrapper does not certify those merely by returning a mode label. Ordinary
raw/legacy operations are still explicitly identified as `LEGACY`.

## Remaining work before an all-phase production claim

The standalone BC trainer currently binds request-specific state at class import,
uses `/dagger-runtime` output/observer paths and has its own fresh-trainer,
source/capacity/storage admission and terminal manifest lifecycle. It cannot be
attached to the DAgger trainer by calling `collect_round` or changing a mode
flag. A native BC adapter must preserve those checks with per-job construction
and output isolation, then be wired and tested through real warm transitions.

Collection-bound repeated teacher/student evaluation also needs explicit native
adapters (the paired teacher smoke currently waits after completion). Do not
substitute a different benchmark. After these are implemented, deploy a fresh
pinned release only at verified idle and qualify BC→test, training→test→training,
both RTX owners, rendered round trips and actual terminal/checkpoint evidence.
Neither CPU tests nor these source changes authorize reuse of an uncertain
native state or release of any compute.

## Checks run for this change

- Local CPU regression: **139 passed** across the new modes, resident service,
  native-boundary method tests, reset/event state, standalone BC collector,
  teacher/student selection and generated-observation routing.
- Leo canonical source: **48 passed** for the modes and resident-service suites;
  the three production Python files match the locally tested SHA256 hashes.
- Leo's `source-tools-20261024` environment lacks PyTorch. The boundary-method
  suite could not import there; it passed in the local test environment. No
  dependency was installed into a live Isaac container to conceal that gap.
- No native workload, owner replacement, renderer session or allocation change
  was performed. The live all-phase transition qualification remains pending.
