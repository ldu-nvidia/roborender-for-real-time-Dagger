# RoboRender production contract

Revision 10, October 26, 2026 (UTC). This is the active entrypoint. The
[superseded contract](HISTORICAL_PIPELINE_CONTRACT_20261015.md) preserves history,
not alternative production defaults. Immutable manifests and actual receipts
determine what executed; documentation is not runtime certification.

## Current decisions and rationale — authoritative summary

**October28 revised next smoke — mixed actions, Image3F:** the user explicitly
selected online Image3F (offline BC rendering occurs before BC training), and
changed the requested test to50% teacher/50% student actions and2 complete
episodes per lane. Interpret the stated64 total environments as32 per RTX on
both actors,128 scored outcomes total; confirm that interpretation in the
handoff. Use an independent seeded Bernoulli draw per lane per vector step,
persist actual selected actions and selection masks, and render/run the frozen untrained student
on every lane before selection. Homie supplies the lower-body actions. No
optimizer, teacher50 online substitution or chunk cutoff. This is a separately
identified mixed-policy infrastructure diagnostic, NOT pure-student success
and NOT a DAgger training update. Preserve the pure-student contract below for
later policy evaluation, but do not launch its3-episode plan for this request.
Mixed-policy integration passed193 focused CPU tests and native-container
configuration/checkpoint preflight. The user approved replacement of the two
verified-idle teacher owners; these were stopped with results/compute retained.
The pinned replacement run `mixed32-image3f-20261028-v3` is starting under
source SHA `8d3038d90ed6ddd5b1e89f91ad9b2182335272e9c2704f46611914dbb4c0ae80`.
The prior v2 startup failed before evaluation admission: a nested Hydra override
destroyed inherited domain-randomization settings. The correction changes the
global block and verifies full resolved inheritance in the native-image CPU
preflight. Actual rendering/128-outcome completion is not yet certified.
Per-step traces contain teacher/student/selected actions, branches and outcomes,
not complete simulator-state snapshots; they do not authorize replay. Only checkpoint policy weights load:
saved optimizer/curriculum/RNG metadata is explicitly ignored.

**October28 collection-bound student-test contract:** the next diagnostic uses
the preserved initial, untrained student (SHA256
`720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a`), never a
retired BC checkpoint. Each RTX uses the same ordered32 motion conditions from
BC run `bc-production10000-20261027-074500`, retaining the entire295-file reset
inventory. Two actors therefore mean64 concurrent lanes, not64 distinct motions.
Each lane contributes its first3 naturally completed episodes:96 outcomes per
actor,192 total. Keep failures; ignore additional fast-lane completions. A safety
stop is an incomplete test, not a reason to manufacture missing outcomes.

The canonical `bc_initial_conditions_32` artifact in `configs/storage_paths.json`
preserves the original request bytes, their SHA256, collection seed, ordered
lane/motion/file hashes and full reset inventory. This is motion/reset support,
**not an exact historical simulator/RNG snapshot**. Draw fresh reset states from
that support with explicit independent actor seeds. Keep corrected initial
robot/object refresh and full native episode clocks; do not reintroduce the
pooled benchmark's shortened initial clocks or historical fixed295 cases.

`STUDENT_TEST` must install the exact frozen checkpoint, clear previous phase's
policy/Homie/episode/action histories and renderer sessions at a verified idle
boundary, and preserve Kit. Capture synchronized current three-view depth and
textured robot RGB controls, render all32 lanes, normalize generated
head/left/right views once, infer student11-D means, append Homie15-D actions,
then execute exactly one native batch step. No native-image policy fallback,
teacher takeover, optimizer/RMS/BatchNorm update, future-state render condition,
partial-batch step or81-row/window reset. Natural done/timeout resets only the
affected lane's episode histories and appearance anchor. Count the native
pre-reset goal OR across each complete episode, not action MSE. Return both
owners to same-Kit physics-IDLE and close per-test rendering sessions. Retain
all compute. Aggregate actual successes divided by actual scored completions;
label partial results until all192 exist.

Renderer selection must be explicit and hash-bound, not guessed from a round
number: `offline_bc_teacher50` is the stored-clip BC renderer; `online_image3f5`
is the existing causal DAgger/student-test renderer. Their weights, slot orders
and conditioning differ. The user mentioned choosing either; a teacher50
closed-loop test needs a separately validated causal adapter and cannot use
future trajectory frames. No automatic substitution is allowed. The current
CPU student plan has `ready:false` and is not native launch admission.

**October28 actual teacher-mode qualification:** frozen source `4bbfb9f52cdb...`
completed paired32x3 tests:87/96 on each RTX,174/192 (90.625%) combined.
Independent outcome/transition checks passed; IDLE→TEACHER_TEST→IDLE retained
each Kit/PID and duplicate delivery did not replay. Evidence:
`runs/teacher32-mode-three-20261028-v1` on Leo. Both retained owners implement
only IDLE/TEACHER_TEST and consumed their one evaluation admission. This does
not qualify the new student adapter, rendered B32x2 capacity, or arbitrary phase
reuse. Their frozen source must not be hotpatched to add STUDENT_TEST.

**October28 explicit phase interface (source/CPU implementation, not live
certification):** `run_phase` names `IDLE`, `BC_COLLECT`, `TEACHER_TEST`,
`STUDENT_TEST` or `DAGGER_TRAIN` while preserving each operation's native
admission, identity and scientific protocol. BC rendering/training remain
offline H200 operations with Isaac idle. The selector must not choose fixed295
implicitly, bypass generated-only inputs, relabel legacy two-window DAgger,
reset unfinished work or release compute. Successful completion requires
mode-correct evidence, synchronized CUDA, paused physics and unchanged Kit;
uncertainty blocks later phases without replay. Resident BC collection and
collection-bound repeated teacher/student adapters are still unfinished;
`BC_COLLECT` is explicitly unavailable rather than mapped to DAgger collection.
See [implementation status](../pipeline/deployment/PHASE_MODES.md). Existing
warm one-shot smoke processes are not upgraded by editing source.

**October28 warm-runtime reaffirmation:** keep both Leo Isaac runtimes alive
and physics-idle between compatible runs; normal run completion must not tear
down Kit or release any compute. Immediate reuse requires an actual resident
command interface and verified idle, episode/history/RNG and dataset/policy
transition boundaries—not merely a process waiting after a smoke test. The
current collection-bound teacher smoke retains its process but is one-shot;
do not describe it as reusable for arbitrary subsequent runs. Finish accepted
paired qualification unchanged. A necessary source upgrade uses a fresh pinned
lifecycle at a verified safe boundary, never a hot patch or replay of unknown
simulator state. No artificial activity to keep an allocation busy.

**October27 production admission implementation (not yet live certified):**
the oneRTX32/10000 natural-drain request has a distinct marker and native ACK.
It cannot be relabeled qualification. Before a production launch, the host must
rebuild independent full-buffer capacity and actual natural-drain execution
proofs, bound to identical source, models, image, GPU and memory contract. The
drain test must cross a real recorded terminal and exclude new-episode rows.
The immutable launch binds both proofs, request bytes and rolling disk policy.
Legacy whole-campaign budgets remain unchanged; this explicit protocol uses
81GiB rolling-window admission, per-archive uncompressed size bounds and64GiB
reserve. It does not guarantee whole-campaign fit; shortage is an incomplete
failure, not target success. Production native and host checks reject all
memory.max/OOM/swap events. Completed-output ingestion verifies actual accepted
windows, drain closure, excluded rows and proof chains; qualification/retired
data remain excluded. Source tests are not native execution certificates.

**October26 Pacific unattended BC goal (no final time limit):** finish and
independently inspect the paired32+32 smoke, safely fix/retest faults, then
collect production on **one RTX with32 environments**, targeting at least10,000
complete81-row chunks. The user's example used64 in its repeat estimate, but
the explicit production request is one32-lane scene; use32 as the denominator.
The requested final boundary is a new versioned natural-episode drain: after
the total target at a committed window, stop recording newly reset episodes
and finish those already being recorded. Overshoot is allowed. Keep Raven's
ordinary reset/action/RNG/chunk logic and short terminal-tail discard. Native
vector lanes still step together; post-drain unrecorded rows are accounted
explicitly, never represented as completed training data. This stopping rule
is a user-requested extension to Raven's finite-window budget, not verbatim
Raven. Draft CPU implementation/tests are not live qualification or admission.
Storage/capacity/complete-output verification remain mandatory; do not silently
waive old budgets, restore retired data, hotpatch live owners or release compute.
Work continues beyond eight hours toward the goal or a concrete required user
decision. Current handoff: `runtime/bc-production10000-20261027/HANDOFF.md`.

**October27 draft functional-drain/disk qualification:** a small total-target
probe is limited to at most16 windows and target1..2B. It is separate from
full-buffer capacity and cannot produce its capacity certificate. The new
versioned streaming disk guard retains512MiB per possible emission,64GiB free
reserve and1GiB metadata (81GiB admission for the nextB32 window). It bounds
each actual archive from uncompressed array sizes before writing and verifies
the result; compressed averages are not admission bounds. Whole-campaign fit
is not guaranteed, and disk shortage remains an incomplete/failure outcome.
These source/CPU changes still require actual frozen-source qualification.
Single bounded launches now share exact scope/PID observation and cooperative
step/writer-boundary holds during bounded transient telemetry recovery. No
state replay, host-policy rewrite or relaxed OOM/swap rule is authorized.

**October27 BC scope-policy repair:** paired060100 completed GPU0's four
windows/91chunks, but GPU1 stopped during final saving after an unlimited kernel
limit produced an integer-parser error. The prior observer did not record the
field, and the host journal is inaccessible, so the exact host-side trigger is
not proven. A real CPU-only probe confirms BC's equal Docker memory/swap flags
leave systemd MemorySwapMax=infinity despite initial kernel swap.max=0. The
resident launcher already handles this drift hazard. New BC launches persist
MemorySwapMax=0 at scope creation and attest systemd+kernel limits, without
changing global host policy. Unlimited limits now produce named durable policy
faults, not unknown parsing retries.279 focused tests and the CPU scope probe
passed. Paired065000 subsequently saved91+90chunks/all windows, with zero unsafe
memory events, but its controller failed during native-exit/Docker-client-lag
reconciliation. The original FAIL remains; GPU1 cleanup sentSIGTERM before its
Docker exit0, so this is not an unassisted end-to-end PASS. New source waits only
after validating the exact terminal receipt, requires launcher exit0 and checks
the terminal bytes again. No dead-PID/receipt-only shortcut.194 focused tests
pass. A real128MiB CPU-only Docker reproducer passed the disappearing-PID plus
delayed-client boundary, with no GPU/host-policy changes. Fresh paired retest
is still required. Staged
drain072000 contains the old race and must not launch. All compute/results stay.

**October26 Pacific paired B32 smoke authorization:** the user requests32
environments on each Leo RTX and explicitly permits a4GiB host reserve for
this bounded test. The paired diagnostic runs four81-step windows on GPUs0/1,
seeds0/1, one initial reset per scene and native done/timeout continuity. It
retains the full ordered motion inventory; both scenes use its first32 entries,
not64 different motion assignments. No rollout/chunk selection changes.
`test/pipeline/run_paired_bc_smoke.py` owns a fresh detached pair and the same
exclusive host-controller lock; `paired_bc_guard.py` uses shared capture/step
leases and an exclusive CPU-save lease. The peer remains warm while saving.
Actual shared MemAvailable must retain4GiB plus projected scratch/tail and peer
capture reserves; individual43GiB caps,2GiB cgroup margin, GPU floors and zero
owned swap/OOM/limit events remain. Initial52GiB availability is a diagnostic
startup allowance, not proven capacity or a summed-cgroup guarantee. Single
collector defaults remain8GiB. No privileged host changes, native replay,
production dataset admission, H200 activity, recurring watchdog or compute
release. Live attempt `bc-pair32-20261027-050100` stopped on an observer
rejection after2/3accepted windows; the rejecting sample was lost by a logging
bug, so the cause is UNKNOWN and the pair is NOT certified. Sampled host
availability stayed at least11.626GiB; retained samples had no owned swap/OOM/
limit events. Both exact workloads stopped; compute and partial outputs retained.
The repaired observer binds exact Docker/PID/cgroup identity, reads counters
directly, and persists every observation/rejection before raising. Transient
unavailable reads hold native step/save-lane boundaries and allow bounded
read-only recovery; no reset, resubmission, stale-sample admission or pressure
waiver. Actual kernel memory/no-swap limits are checked.205 focused CPU tests
and a CPU-only Docker fault-injection probe pass. Docker events show controller
stop signals and no OOM event for the failed pair; the lost rejecting sample
still prevents identification of its exact subcondition. No B32x2 native rerun
has passed; observer validation does not retroactively qualify that attempt.

**October27 B40 smoke PASS:** the frozen cachefix release
`6e8f069b4cd821706f25e726cc23b8fd49a6d41373d2523de3db16f4a3c1b370`
completed one RTX0 scene with40 environments,27 windows/2187 vector steps,
854 complete chunks (20–23 per lane), one initial reset_all and zero optimizer
steps. Exact run: `runs/bc-continuous/smoke40-cachefix-20261026-231200` on Leo.
Terminal native and host receipts passed; an independent read-only audit verified
every payload/provenance hash and81-row array header. Host cgroup peak35.688GiB
of43GiB; GPU peak67.692GiB of95.593GiB (70.813%); zero memory-limit/swap/OOM
events.151 focused CPU tests passed. All compute retained; only this completed
standalone workload exited. This qualifies the requested bounded B40 smoke, not
an unlimited campaign, two concurrent collectors, B50, a production dataset,
or the separate4/8-window capacity certificate. Production admission remains
false; no historical failure is relabeled or resumed. Completed outputs and
verification evidence remain in the run/stage directories, not source.

**October26 B40 save-memory recovery:** the first B40 attempt failed during
first-window saving at41.08/43GiB host cgroup usage (no recorded OOM/swap),
13 uncommitted NPZ files, no accepted window. Preserve that failed namespace.
The user authorizes a tested writer-memory repair and fresh B40/20-chunks-per-lane
smoke on one RTX. Explicit per-lane scratch lifetimes, unused-heap compaction
and local memory telemetry do not change data/episode/RNG semantics. Retain
the same43GiB cap, host/device floors and diagnostic-only publication boundary.
No relaunch/replay of the failed simulator; no H200 work or compute release.
Actual full-run evidence is required before declaring40 lanes successful.
The first repair committed two windows/54 chunks, then the third save was
rejected by projected reserve (32GiB current including17.4GiB file cache;
zero OOM/swap). It is terminal, not resumable. The next source also fsyncs
each exact owned completed NPZ and requests POSIX_FADV_DONTNEED after writing
and provenance indexing. This is advisory cache release, not deletion or a
relaxed admission estimate: actual counters must still pass unchanged guards.
No global drop_caches, broad cache traversal or historical-source patching.

**October26 B40 follow-up:** after the B50 smoke failed its GPU headroom guard
at72 vector steps with zero complete chunks, the user authorized a fresh40-lane
attempt on RTX0. Extend only the explicit chunk-target diagnostic to B40 or B50;
retain20 complete chunks per lane, the40-window/12-hour safety ceiling, original
host/device floors and all-emissions behavior. Do not replay/relabel the B50 run
or call either width certified. No compute release or H200 work is authorized.

**October26 explicit B50 smoke:** the user now requests one50-environment RTX
collector until all lanes have saved20 complete81-row chunks. This separate
`roborender.bc-all-lanes-chunk-target-smoke.v1` diagnostic stops at a committed
full-window boundary, preserves extra emissions from faster lanes and does not
pause/reset individual lanes at quota. A40-window ceiling/12-hour timeout fails
incomplete if any lane misses the target. It is not the original20-window test
or a change to ordinary continuous Raven collection. B50 is experimental; retain
host/device guards and exclude diagnostic data from production. Completion does
not produce a capacity certificate. No H200 work or compute release is needed.

**October26 BC memory-observer correction:** persist and fsync rejecting samples
before cleanup; record distinct swap/OOM/limit-event reasons and native-gate
snapshots. Only the qualification-only GPU-window BC profile may observe a
recovered cumulative `memory.events.max` event as a warning, with at least2GiB
current cgroup headroom and peak no greater than the unchanged40/43GiB cap.
Keep zero swap/OOM, the8GiB host floor and native allocation projections. This
permits diagnostic evidence gathering, not a capacity certificate: zero limit
events and complete repeated windows remain required for certification. Other
profiles remain strict. The failed first B32 probe's triggering sample was not
persisted; do not reconstruct it or claim a confirmed OOM. Frozen source and
receipts remain untouched; no collection restart follows from this code edit.

**October26 standalone BC runtime organization:** the collection launcher owns
admission and lifecycle; a shared `rtx_actor/bc_runtime.py` owns mount assembly
and canonical path resolution. New BC uses the pinned central registry, physical
`simulation_assets`, `caches/isaac` and `runs/bc-continuous` under the Leo data1
project root, and saves a per-run path snapshot. No active BC home/data2 fallback,
scientific algorithm change, frozen-owner patch or new capacity certification
is introduced. Historical consumers keep explicit compatibility behavior.
Docker data-root remains `/var/lib/docker` pending administrator maintenance;
source refactoring does not imply that system-store migration was performed.

**October26 standalone BC collection:** new plans bind
`roborender.bc-standalone-collector.v1`. `pipeline/bc_collection/collector.py`
owns the single-reset continuous vector loop and `chunk_writer.py` owns the
extracted Raven writer. Default32 lanes repeatedly follow natural done/timeout
resets; no per-motion8/9-chunk quota or window-boundary reset. Preserve full
motion inventory/order, randomized initial clocks, pre-action conditioning,
teacher/Homie actions,81-frame append→emit→done-tail ordering and incidental RNG.
The native adapter now accepts the original inclusive sample-range endpoints.
No learner, validation, scheduler or checkpoint-saving callbacks execute in the
new loop. Shared Isaac/policy construction and reset/provenance/resource guards
remain integration dependencies. Old protocol-absent requests remain explicit
compatibility only. Freeze/mount/hash both new modules in execution receipts;
CPU differential evidence is not current-source native B32 capacity admission.
The overall window budget can end mid-episode; no final-tail padding/flush or
guaranteed equal episode repeat count is introduced. No live owner upgrade,
production collection, watchdog restart or compute release accompanies this edit.

This section controls new work; older dated recipes below describe their own
lineage. The RoboRender skill routes here rather than defining another protocol.
Compare against pinned Raven commit `94d79851d9c471b1210be81c75116d9bea835a4d`;
the [source audit](../pipeline/RAVEN_EQUIVALENCE_AUDIT.md) records exact comparisons
and unresolved differences. Do not claim complete functional identity.

**October26 cyclic/reset repair:** new fixed-final BC uses persistent sorted
cyclic full batches, wrapping inside a batch without resetting the cursor at
reporting epochs. A reporting epoch is `ceil(N/B)` updates; save/resume commits
only completed-update positions, never prefetch. New continuous eight-H200
DAgger configs require `reset_inventory_protocol=roborender.raven-full-reset-inventory.v1`:
both actors retain the complete ordered295 reset inventory, with global sampler
lane offsets0/16 and unchanged local physics lanes. Full inventory/hash/offset
receipts are mandatory; never shrink or reorder that loader. This restores
Raven's full-inventory resampling, and the observer accepts its inclusive final
sample index. Frozen old jobs are not upgraded in place. CPU differential and
resume evidence is in `runtime/raven-parity-overnight-20261026-t6oMpN`; live
continuity and warm train↔test certification must be recorded separately.

**October26 warm-event correction:** an independent phase must restore both
`common_step_counter` and the same scene's cold EventManager reset-trigger
counters. Otherwise the previous phase's timestamps can suppress material/light
randomization. Copy the actual cold lane tensors without RNG draws or physics
steps; do not call `EventManager.reset()` as a substitute. Reject unsupported
startup/interval/class-event state. Never apply this restoration at a learner
window or ordinary natural episode reset. This matches fresh-task reset-event
eligibility; it does not promise bitwise Isaac5/Isaac6 or renderer identity.
The exact Lab3 `apply` regression passes; new-source native transition evidence
is required separately, never inferred from the old owner's capacity trials.

**October26 live evidence scope:** source `affaf6e3...` completed paired rendered
B4/B8/B16/repeat16 and tail7 with actual resource/W&B receipts. Its fresh63/B32
H200 BC probe also verified cross-pool cyclic indices and exact model/AdamW/RNG
resume (four reference plus two resumed optimizer operations). These are bounded
diagnostics, not a production BC policy, full295 result, full-buffer BC collection
capacity or a new-source warm transition certificate. See the overnight HANDOFF.

**October26 BC collection RNG:** new continuous plans bind
`roborender.bc-collection-incidental-rng.v1`. At each original learner boundary,
consume Raven's full-batch `randperm(B)` then discarded `Normal(B,81,11)` draw,
without optimizing on native RGB. Enforce pinned source/architecture/dtype
guards and the emitted window-count receipt. This corrects the reset-stream
change caused by omitting Raven's incidental learner. CPU differential tests
pass. The separate actual-device differential also passes on both RTX GPUs:
B1/2/16/32/40, three seeds and five windows each, matching the original Normal
draw stream and next-reset draws. It leaves both native owners untouched and
does not certify a new live physics/collection trajectory. Full-buffer collection
capacity remains separate. Never retrofit this onto an accepted frozen owner.

**October26 bounded BC storage qualification:** the original full-B32 CPU
history and save allocation cannot fit the existing43GiB host limit. The new
explicit `roborender.bc-gpu-window-lane-writer.v1` profile keeps the raw81-row
window on the admitted RTX and converts one lane at a time on CPU; it does not
split the shared B32 scene/curriculum. Normalize each original window before
within-episode reassembly, retain append→emit→done ordering, and make tails own
their bytes. Reserve the complete raw GPU history, one-lane stack and device
floor; account for copied-tail high-water separately from conversion scratch.
The original default and guards remain unchanged. This is qualification-only,
explicitly reports a changed writer, and cannot publish production data.
351 focused CPU tests passed, including pinned-original capture/writer hooks,
NPZ byte comparisons and RNG/reset continuity. Full-resolution recorded-input
comparison, actual CUDA transfer and live repeated B32 capacity remain separate
gates; CPU success is not their certification.

**October26 V6 bounded DAgger evidence:** source `6052af310ef5...` completed four
genuine full32×81 updates with one warm native initialization per actor, shared
curriculum, full295 reset inventory and78 preserved AdamW states. MSE values were
1.4094721,1.1113119,1.0423043,0.9180398; actual W&B readback and whole-interval
resource checks passed. This proves that bounded continuous training leg, not a
100-update production round or the remaining warm train/test round-trip. The
user removed the eight-hour horizon; renewed source-bound candidate evaluation
and remaining certification continue under `user-authorization-resumed.json`.
Historical admissions and failed attempts remain unchanged.

**October26 learner placement:** pin the full32×81 learner to one admitted H200
UUID; retain all8 independent renderer/gateway UUID mappings. Actual process
visibility must match before dispatch. The live probe exposed inherited8-device
visibility and failed before model construction; this was not OOM. A separate
saved-batch single-device retry completed with78 AdamW states and~49.6GiB free
GPU memory. It does not certify native continuity; failed partial trajectories
remain excluded and fresh-owner qualification is required.

**October26 teacher context hooks:** match Raven's distillation wrapper for
optional teacher `init_rollout`/`clear_rollout` exceptions, while propagating
student and environment failures. The deterministic teacher does not recreate
the optional distribution field deleted by its first clear; subsequent clears
must not kill collection or trigger a simulator reset. Clear/begin remains an
81-step model-context boundary, never a physical episode boundary. Regression
uses the actual pinned Actor methods; live multi-window proof remains separate.

**Latest October25 priority — ordered functional certification, not refactoring:**
compare BC collection → paired collection smoke → compare success calculation
→ paired same-input success smoke → compare DAgger → paired DAgger functional
smoke. Keep generated-only student inputs and corrected reset explicit; compare
identical declared configuration/input/model identities, not unrelated scores.
Do not promote a CPU or outcome-replay comparison to live physics/renderer/learner
certification. Current step1/2 evidence is in
`runtime/raven-stage-certification-20261025-kBFQ3q/status.json`:176 prelaunch BC CPU
tests passed. The authorized B2 native smoke completed4windows/324vector steps,
five chunks and one initial reset. Replaying its exact captured writer inputs
through pinned Raven produced identical NPZ bytes for all five chunks; no second
Isaac trajectory was replayed. Peak cgroup27.70/40GiB, zero OOM/limit/swap events.
A test-only camera dictionary-order fix passed23 regression tests; native source
and evidence were unchanged. This is bounded B2 evidence, not B32 or long-run
admission. Success-calculation comparison has76 CPU tests passing. A separate
actual historical teacher-trace replay covers all19cohorts/295cases: pinned
Raven and current first-episode scoring agree on255successes/40failures and all
per-case recorded rewards/lengths. This is not a second native rollout or Raven's
default pooled protocol; full trace hashes are newly observed, not originally
sealed. Both Isaac owners are restored authenticated IDLE, checkpoint=null,
without releasing compute. Matched live evaluation and DAgger live equivalence
remain pending. The coupled DAgger CPU comparison passes69 combined tests,
including15 new conditional B4/three-window cases with natural terminations,
continuity beyond162, exact data/labels and strict78-slot AdamW moment checks.
It uses scripted physics and a tiny network, an explicit behavior-selection
tape and declared numeric tolerances; RNG, sorted-vs-permuted learner order and
independent B16 curriculum differences remain. It is not full functional or
native certification. The unchanged B32
full-buffer save guard cannot admit its mandatory depth histories; do not weaken
it or substitute evaluator capacity. Later live stages have not been certified.
Existing datasets remain retired and all compute remains retained.

All first-party unit/regression/functional tests and test-only helpers belong
under the single repository-root `test/`; production must not import them or
test frameworks. Use `bash test/run_cpu_checks.sh` for the CPU gate.
Pinned upstream references and materialized external dependencies stay unchanged.
Keep runtime safety/admission checks in production and immutable old receipts
unchanged. See [cleanup boundaries](CODE_CLEANUP_20261025.md).

**October25 BC retirement override — takes precedence over BC reuse/launch
instructions below:** the user instructed, "delete all our currently collected
BC training data and all the bc round checkpoints". These datasets and trained
BC checkpoint lineages are retired from future use; active registry selectors
for BC116, clean collections/splits and merged2880 are removed. The scoped
authority, inventory, recoverable removal status and restoration map are in
`runtime/bc-retirement-20261025/RETIREMENT.json`. Registry removal is not proof
that every payload has moved; only that record's verified execution receipts
establish completion. Preserve provenance, historical results/W&B, initial
student bootstrap, teacher/Homie, renderers/anchors, motions and DAgger assets.
No automatic restore, fallback to an old BC pool/checkpoint, recollection or
training is authorized by a missing-file alert or watchdog. Fresh collection/
training requires a new explicit user request and new admission; restoring
retired payloads requires explicit user authorization. Do not patch frozen
owners or release any Cosmos compute as part of retirement.

**October25 continuous BC collection override:** new collection uses
`pipeline.bc_collection.continuous_plan` and `pipeline.bc_collection.run_continuous`,
not the retired coverage/quota campaign. One shared vector scene (reference
default B32), seed0 unless explicitly bound otherwise, retains its fixed lane
motions, reset sampler and success-dependent curriculum for the whole run.
Record the full ordered eligible motion inventory, including unassigned files
loaded/resampled by Raven; never sort/hash-shuffle it or rotate motions between
windows. Historical filesystem ordering is unknown unless supplied as evidence.
Two independent B16 scenes are not silently substituted for one B32 population.

Require an explicit budget of81-step vector windows; quantity may increase,
but no per-motion quota, chunk-ordinal selection, train/validation partition,
success filter or intermediate physical reset is allowed. Keep every emitted
complete within-episode81-row chunk, append→emit→done ordering, native short
terminal-tail discard and nonterminal carryover. Never flush/pad final partial
buffers or promise every episode finishes before the finite budget. Original
three-view pre-action conditioning, labels, window-based depth normalization,
50Hz control and natural resets remain unchanged. One completed native lifetime,
exact budget, all-emissions index and execution evidence are required for pool
publication; uncertain/partial simulator state is never replayed.

Keep corrected initialization, Isaac6 portability and zero-optimizer collection
explicit: the original collector's incidental native-image learner updates are
omitted to honor generated-only student training. New protocol-bound collection
preserves its discarded random draws, not its unused student weights. Distributed
RNG and Isaac-version numerical identity are not established. No held-out
collection/validation is reintroduced. The new code is **not live-certified**
until genuine current-source full-buffer BC capacity and multi-window lifecycle
tests pass. B16+16 evaluation/DAgger evidence is not BC-collection admission.
The present user request authorizes implementation/testing, not restoration of
retired data, an unbounded collection launch or compute release.

| Decision for new runs | Reason / relationship to Raven |
|---|---|
| Central `configs/storage_paths.json`; separate source, immutable releases, assets and run evidence on each host | Operational organization only; changing a path must not change model/data identity or algorithm. See [layout](STORAGE_LAYOUT.md). |
| No held-out MSE validation or patience selection in BC or DAgger; keep integrity checks and stage-end success tests | Removes our added validation-selection loop, which Raven's supplied training does not use. Our explicit BC epoch budget and four fixed100 mixed-policy rounds remain deliberate experiment choices. |
| Previous BC clean2560+former320 union:2880 chunks/233280 frames is retired; no current default pool | Preserve original provenance and never call those320 held out again. The retirement override supersedes earlier reuse instructions; a future pool needs fresh admission. |
| Qualify BC from the saved initial student bootstrap with one full32×81 update population, not BC116 continuation or two accumulated16×81 microbatches | Requested scale increase. Preserve Raven's81 no-grad replay forwards, permutation, full learning forward, BatchNorm lifecycle,11-D action MSE and AdamW. Larger batches change statistics; numerical identity to Raven's B16 is not asserted. |
| DAgger uses one scene per actor per round; no forced reset after one/two chunks or at the four-update qualification gate | Raven continues returned observations between81-step model-buffer windows. The old two-window cohort scheduler discarded unfinished physical trajectories; remove that scheduler behavior, not natural done/timeout lane resets. |
| Preserve fixed lane motion assignments and natural episode resets; collect all lanes'81 rows before one joint32×81 update | Matches Raven's temporal collection/update ordering. Lanes may be at different episode ages, and a chunk may span a real episode end. A chunk boundary is not an episode boundary. No prior-window replay. |
| Generated three-view student input only: offline teacher50 for BC, online Image3F5 for DAgger and success testing | Explicit user exception, not verbatim Raven: avoid direct native-RGB policy inputs. Native RGB/depth may condition the renderer only; rendering failure must not fall back to ray tracing. Offline/online renderer differences still require honest reporting. |
| Correct initial robot/object refresh consistently in collection, training and testing | Retain the diagnosed initialization correction; matching Raven does not authorize restoring the first-step robot/object mismatch. Preserve ordinary reset and native goal-success rules. |
| Keep both Isaac Kit processes warm, with explicit committed-checkpoint/IDLE/reset barriers at train↔test transitions | Removes repeated Kit startup overhead without carrying training simulator state into the independent test. Testing uses frozen student+Homie, zero optimizer steps, and native goal success—not an MSE threshold. |

The required online topology is **RTX GPU0:16 lanes → H200 slots0–3;
RTX GPU1:16 lanes → H200 slots4–7**. The two four-worker sets must contain eight
distinct admitted GPU UUIDs, with no shared gateway/renderer aliases. The two
RTX actors may progress concurrently; all32 lanes join the learner barrier for
each81-step window. One optimizer update has2592 transition rows. Training
student inference runs on H200; the scaled tester currently runs frozen student
inference on RTX and renders on H200. No validation stage is inserted.
Offline BC uses32 stored chunks/update on its admitted H200 learner; it does
not step32 live Isaac environments. The merged pool has90 such updates/epoch.

### Consented differences from Raven — October25 user confirmation

The user explicitly asked to preserve the agreed differences below as
**consented modifications**, not anomalies merely because they differ from
Raven. Future audits must separate consent, implementation correctness and live
certification. Do not silently revert these choices during parity work; consent
does not certify execution, guarantee identical results or waive resource gates.

| Consented modification | Motivation and boundary |
|---|---|
| Generated-only student inputs in training and success testing | Avoid native-ray-traced versus generated policy-input mismatch; learn from the generated observations used for prediction. Native images may condition rendering only. Offline teacher50 and online Image3F5 remain distinct distributions. |
| Three views and increased environment/learning batch sizes | Retain head/left/right perception and increase throughput on the requested RTX/H200 topology. Raven supplied a three-view extension; published single-view results are not an identical experiment. Larger batches change normalization statistics; scale approval does not approve independent curriculum behavior or establish capacity. |
| Correct robot/object initialization and preserve terminal-step goal accounting | Retain the intended task state and count genuine success events instead of reproducing known initialization/export defects. Do not change the task's goal predicate. |
| Four mixed-policy DAgger rounds,100 accepted updates each, final checkpoint selection | User-requested teacher probabilities75/50/25/0 expose student-visited states under teacher labels with a simple fixed budget. BC uses an explicitly supplied epoch budget. No held-out MSE/patience stage; success testing remains separate and never selects checkpoints. These budgets/mixtures are not Raven's recorded experiment. |
| Fixed295 retained alongside a separately versioned Raven-style success benchmark | Preserve within-protocol checkpoint comparisons while allowing evaluation sampling closer to Raven. Never merge their outcomes, relabel old scores or silently replace the selected protocol. |
| Warm Isaac and distributed RTX/H200 execution with durable recovery records | Avoid repeated startup overhead and improve throughput/recovery while preserving action/step/update ordering and clean train/test isolation. An accepted request must not cause duplicate physics steps. Warm handoff still requires actual live proof. |

The **Raven-style benchmark calculates closed-loop task success**, not training
or validation MSE: freeze the student, randomize initial episode/stage clocks
as in Raven, pool completed episodes, and divide successful completed episodes
by all completed episodes, including failures and final-vector-step overshoot.
It is closer to Raven's original sampling than fixed295's first-episode/fresh-clock
protocol, not an exact historical reproduction. The implemented scaled form is
two independent B16 replicates, each targeting at least150 completed episodes;
the combined rate is total successes / total completions, not one global B32
pool. Generated inputs and corrected initialization/accounting remain required.
See [the separate benchmark contract](#separate-raven-style-success-benchmark).
Its implementation/CPU evidence is not live certification; this confirmation
neither launches it nor changes the current evaluation selection.

**Not covered by this consent:** historical DAgger hash-based versus original RNG draws,
unpermuted learner lanes, independent per-actor curriculum versus a
shared population, and unresolved motion/initialization/cold-curriculum sampling
differences. BC collection's omitted incidental learner/RNG work and BC loader/
bootstrap execution differences remain disclosed parity gaps, not newly approved
algorithm changes. Unknown original optimizer dependencies, Isaac5/6 numerical
physics and unavailable original renderer internals remain evidence gaps, not
equivalence claims. Review/fix or obtain a separate decision for these items;
do not infer their acceptance from approved scaling or generated inputs.

Removing premature chunk-count resets and restoring BC replay/BatchNorm behavior
are parity corrections, not exceptions to retain. Removing our held-out MSE
selection also moves toward Raven. Retired datasets/checkpoints remain retired;
this documentation change authorizes no restoration, production launch or compute
release. Existing immutable results and certification limits remain unchanged.

### Certification required by the October25 follow-up

**October26 source correction (not yet live-certified):** newly generated
continuous eight-H200 configs bind three explicit protocols: native-vector
PyTorch teacher/student draws (including Raven's discarded student Normal draw),
saved-RNG environment `randperm` before each full B32×81 learner update, and
shared B32 curriculum statistics at every paired native reset boundary. Time
within each lane stays chronological. Empty local reset sets still participate
in global curriculum updates, without resetting unfinished lanes. Exact-owner
barriers and saved action choices reject uncertain replay. Old admitted configs
retain their versioned semantics; new-protocol checkpoints cannot silently
downgrade, nor inherit independent-population DAgger curriculum.
The server's chunk-ID sort was only a membership check, not learner ordering;
the actual missing operation was Raven's permutation. See the latest audit
appendix for tests and remaining distributed-RNG/live-physics evidence limits.
Warm phase preparation also restores Raven's cold vector-step clock; population
journals use a separate pool-owned writable mount, never the read-only jobs or
assets mounts. The user approved one idle-owner source upgrade and bounded16x2
test. The retired BC32 diagnostic checkpoint is not an available parent; a fresh
admitted parent is required and retirement must not be bypassed for testing.

**October26 fresh diagnostic parent completed:** the separately authorized
bounded B2 collector retained104 complete81-row clips across64 windows; these
are clips, not104 finished episodes. All64 selected smoke clips received genuine
independent three-view teacher50 renders. Two full32×81 BC updates from the saved
initial bootstrap passed policy/AdamW/RNG reload and online W&B readback (run
`nvidia/dagger/du3oy4d0`). No retired data/checkpoint was restored, no validation
was run, and no production pool/checkpoint was promoted. See
`runtime/raven-dagger-alignment-20261026-7pYLxI/fresh-parent-evidence-v2.json`.
The original controller's final read-only keyed-history query failed; the tested
v2 full-history query verified the same two actual updates without replaying GPU
work. Both upgraded source2054 Isaac owners are warm/IDLE, and all eight H200s
are retained. Current-source B16+16 evaluation/continuous DAgger qualification
remains pending; the fresh-parent result does not certify those stages.

Execute bounded live checks in order: initial-bootstrap B32 BC learning →
rendered B16+16 success testing → continuous B16+16 DAgger. Then verify the
actual learned-checkpoint train→test→train→test sequence on the same warm owners.
For each check bind actual source/config/assets/checkpoint, owner identities,
memory/headroom/OOM/swap observations, durable outputs and W&B readback.
DAgger evidence must show unchanged physical episode continuity across the old
162-step boundary unless a real done/timeout occurred, plus genuine full32×81
learner updates. A bounded four-update diagnostic is not a completed100-update
round or evidence that every episode finishes. A finite round budget can still
end during an episode; it must not trigger extra resets within the round.

**Overall status: not certified.** A live two-update B32 BC smoke from the saved
initial student passed on source800def3343..., with exact optimizer/policy/RNG
reload, online W&B readback and >=16-GiB guarded GPU headroom. This is not a
completed epoch or production checkpoint. The first current-source52d590...
paired B16 rendered trial completed32/32 with unchanged policy and whole-interval
resource coverage: maximum RTX VRAM fraction0.238, minimum host available
25,995,386,880bytes, no OOM/swap/cap-hit. The repeat also completed32/32;
its577 whole-interval samples passed, minimum host available25,580,126,208bytes,
maximum RTX VRAM fraction0.244 and zero cap-hit events. Tail7 and the final
warm-phase check are still running; do not substitute the old incomplete-resource trial.
Eight-H200 live DAgger collection/learning and the actual learned-checkpoint
train→test→train→test sequence remain unqualified. Separate learner/control source
0c24f0e7... preserves78 optimizer slots and post-update RNG across this diagnostic
handoff;885 source-bound CPU tests passed, but those are not GPU certification.
Preserve scoped evidence under
`runtime/live-pipeline-smoke-20261025/HANDOFF.md`; only actual passing receipts
may advance status. The full295 benchmark remains separate from smoke panels.
No new full BC epoch budget or production campaign is inferred from these smoke
requests. Retain all Cosmos compute, including when a workload stops.

**October25 BC dataset union:** promote the clean320 former validation chunks
into a new2,880-chunk train-only index (`bc_train_union_splits` in the storage
registry). Keep original2560/320 indexes, provenance, payloads, BC116 and test
panel immutable. `bc-train-union-admission.v1` must rederive the exact union from
the original production admission; only `fixed_final` accepts it. Former320
must never be reported as held-out validation for this lineage. No retraining
or changed-data checkpoint resume is authorized by the merge. New success
testing still requires the generated-only scaled-evaluation admission; the old
source admission's ray-traced test descriptor remains historical only.

New standard `fixed_final` launches resolve the merged pool from the central
registry; partial dataset/split overrides fail. Fresh BC defaults to versioned
`raven_offline_replay_v1`:81 train/no-grad student forwards, lane permutation,
then one full configured B×81 learning forward, including the zero-weight object-head
optimizer lifecycle. Legacy BC116 retains its old identity; cross-lifecycle
resume fails. B16 is the historical tested setting; B32 is the new separately
admitted certification target. This is not native qualification.

Fresh B16/B32 Raven-replay BC carries all78 AdamW slots, including8 zero-loss
object-head slots, into DAgger. The versioned
`roborender.dagger_replay_student_checkpoint.v1` successor preserves that BC
optimizer lineage across updates/rounds; never discard slots to fit the legacy
loader. Historical BC116's70-slot direct path stays distinct. CPU differential
tests cover100 updates and the next round; live paired integration remains pending.

**October25 parity audit:** rendered student inputs remain mandatory by explicit
user confirmation. Read the updated Raven audit before claiming end-to-end
equivalence: original-server internals, evaluation sampling/clocks and global
curriculum carryover remain unresolved. Student success entrypoints must require
`enforce_teacher_rollout=false`, `ratio_teacher_rollout=0` and
`rollout_with_teacher_num_steps=0`; original inherited defaults can override
student actions even in eval mode. Keep privileged-teacher diagnostics separate.
Do not silently change the frozen295 benchmark to random initial clocks or
restore training curriculum into test. New behavior needs a new bound protocol.

**October25 DAgger continuity correction:** the eight-H200 target uses
`roborender.dagger-continuous-fixed100-plan.v2` and matching continuous-v2 config/
queue; v1 remains the distinct four-worker topology. Neither uses fifty
two-window native lifecycles. Initialize each Isaac scene once
per round; carry physical observations, episode cursors and anchors through all
100 updates. Native done/timeout alone resets individual lanes. Raven's model
rollout buffers still begin/clear every81 steps; that is not an Isaac reset.
Keep each lane's motion fixed, including natural resets: the explicit first32
inventory-ordered motions are assigned across both RTX actors, identically in
all four rounds. This does **not** cover all295 training motions. Initial/reset
poses remain Raven's reference-frame sampling, not guaranteed task starts.
The100-update end can still leave an episode unfinished; no boundary ever
guarantees success. See the [source comparison](../pipeline/RAVEN_EQUIVALENCE_AUDIT.md).
CPU conformance is not native certification; do not activate this source on an
existing frozen owner. The stopped BC benchmark/scaled-evaluator gates below
remain in force. Historical two-window plans require explicit legacy opt-in.

**Latest warm-phase / no-validation override:** disable held-out MSE validation
and validation-based checkpoint selection for new BC, DAgger and smoke runs.
Keep data-integrity, capacity and success-test checks; do not delete the existing
validation data or relabel historical results. New offline BC uses the explicit
`fixed_final` epoch budget and selects the last complete epoch; no budget is
inferred. The latest request authorizes initial-bootstrap B32 certification,
not an implicit full-epoch budget or replacement of the BC116 baseline.
New DAgger keeps exactly100 accepted updates and selects update100.
Legacy validation execution requires explicit compatibility opt-in, never an
implicit fallback. Keep both RTX Kit processes warm across compatible phases,
with a committed checkpoint / quiescent-owner barrier and an explicit fresh
train/test state reset. See “Training / success-test phase isolation” below.
The eight-worker paired evaluator and training→test→training→test sequence need
separate native evidence; neither source changes nor spare VRAM certify them.

**Latest October24 scaling override:** the user stopped the incomplete BC116
B2 rendered evaluation and requested one complete `custom.gpu.8xh200` node in
the approved reservation. Retain the separate existing one-H200 allocation;
no release, queued-request cancellation, automatic expiry or artificial work.
Build/qualify two Leo Isaac owners at **16 environments each / 32 concurrent
cases**, with four dedicated Image3F workers per actor on the eight-H200 node.
Do not resume the stopped evaluator or automatically start DAgger while this
replacement is being qualified. The older finish295-first instruction below
is superseded by this explicit stop.

Scaled testing retains the ordered295 motion descriptors, BC116 checkpoint,
corrected reset, generated-only input and native goal criterion. Regrouping
Isaac environments changes RNG consumption/initial conditions: use the new
scaled benchmark identity, not historical B2 outcomes. Freeze the new panel,
seeds, batch and rendering contract across later stage comparisons. B16 has
18 full cohorts plus a final7-case cohort; never pad the denominator to304.
Require current-source paired B4/B8/B16 full closed-loop probes, repeated B16
initial-state evidence, a real7-case tail, eight genuine renderer identities,
all-lane barriers and measured host/cgroup/GPU headroom. CPU tests alone are
not production admission. One cohort per native command is the durable stop/
recovery boundary; uncertain partial cohorts are preserved, never replayed.
Implementation: `pipeline/deployment/scaled_success_dispatch.py`,
`pipeline/deployment/rtx_actor/scaled_success_contract.py`,
`pipeline/evaluation/scaled_success_{qualification,resources}.py`.

**Latest October24 user override:** new student training and success testing
must never consume native ray-traced RGB directly. BC consumes the existing
offline teacher50-rendered three-view dataset; DAgger training and student
success tests consume genuine online Image3F5-generated head/left/right RGB.
Isaac RGB/depth remain permitted renderer conditions, not policy observations.
Remove the old50/50 observation-source mixture for this new protocol; missing
rendering fails closed, never falls back to native RGB. This intentionally
changes Raven's historical mixed-observation recipe.

**Earlier October24 instruction (superseded for the stopped B2 test):** finish
the already-running BC epoch116 rendered fixed295 test unchanged, verify
the complete result and online W&B receipt, then end that evaluation workload
at its actual terminal/quiescent boundary. Retain all Cosmos allocations.
Only a fresh admitted source/lifecycle may begin the subsequent DAgger campaign.
The selected starting policy/AdamW is BC epoch116, not the stopped old DAgger.

Each new DAgger round uses **exactly 100 accepted full B32×81 optimizer updates**,
with teacher probabilities 75/50/25/0% in order. There is **no held-out DAgger
collection, validation-MSE pass, baseline check, patience or early stopping**.
Select update100, test it on the corrected rendered fixed295 panel, then carry
that exact policy and AdamW state into the next round. Success does not select
checkpoints or change the 100-update budget. Stop after round 4's test. Historical
BC validation receipts remain unchanged. New source and all-rendered B16×2
native/H200 qualification remain activation gates, not claims of completion.

All new BC and DAgger student success tests, including smoke tests, require
online Image3F-generated head/left/right RGB.
The fixed295 assignments, seeds, ordering, corrected initial reset, Homie and
goal accounting stay unchanged; the rendered panel/protocol has a new identity.
Hash-bind renderer/anchors/checkpoint, require per-step rendering receipts, and
fail before acting if the renderer is missing—never fall back to native RGB.
Historical ray-traced results and frozen running releases remain unchanged and
must not be imported as rendered results. Privileged-teacher diagnostics use
state, not RGB. Image3F5 is not identical to the offline BC teacher50 renderer.
Native activation requires a fresh attested source lifecycle, not a hot patch.

The [clean recollection plan](CLEAN_RECOLLECTION_PLAN_20261020.md) superseded
legacy trajectory reuse and required corrected native reset qualification before
fresh BC training-data collection. Documentation alone was not deployment proof.
That clean collection and BC training have already produced the selected
epoch116 baseline; the current DAgger request does not recollect or retrain BC.

## Hardware, production sizing and ownership

The new success-test target is **two Leo RTX GPUs plus one eight-H200 node**.
The earlier DAgger training target below remains four workers until separately
admitted; the requested four-H200-per-RTX training routing is not yet certified.
Changing routing does not change the algorithm or update budget.
The user requests the highest safe productive throughput, not a smoke-size
production default. **The new candidate is B16 per RTX / B32 total**, not B2.
The full B32 H200 learner passed an epoch116 probe with about 32% projected
device headroom including a 25 GiB renderer allowance. Current-source paired
native and end-to-end qualification are still required before activation.
See [sizing evidence and limits](DAGGER_SCALING_20261022.md). Historical B2 runs
remain immutable. Never infer production safety from spare RTX VRAM alone.

| Work | Placement |
|---|---|
| Orchestration, source, registries and ledgers | CPU access pod, `/workspace/project/roborender` |
| Isaac, privileged teacher, Homie, only `env.step` | Leo physical RTX GPUs 0 and 1; display GPU2 excluded |
| Image3F + frozen student inference | Four reserved H200 workers |
| Full-batch student optimizer | One admitted H200 at the window barrier |
| Offline BC learner | One H200; no live RTX, Isaac or renderer |
| Scaled stage-end success test (qualification pending) | 2×RTX Isaac/student/Homie atB16; 8 independent H200 Image3F workers; no optimizer or teacher takeover |

Leo has roughly 62 GiB host RAM. Qualification must measure full-phase host and
cgroup working sets, swap/OOM/pressure, GPU peaks, full learner batch and wall
throughput. Capture-only tests and unused VRAM are insufficient. Existing B2
owners use 21 GiB caps, no owned swap/OOM and an 8 GiB host reserve; larger
profiles need a jointly admitted host budget. Offline collection's separate
B6/B1 single-collector evidence does not qualify two online B6 actors.

Use approved `cosmos-solution-architect-xcwtffiq` / `custom.gpu.*xh200` capacity.
Never release, cancel, scale down or terminate requested/allocated Cosmos
compute without resource-specific user authorization. Stopping a workload is
not release approval. Do not fabricate GPU activity to evade cluster policy.

## Source and assets

The October24 incremental [storage layout](STORAGE_LAYOUT.md) introduces
`configs/storage_paths.json` for new host-specific locations and frozen per-run
path resolutions. New Leo source lives at `/data1/workspaces/roborender/source`;
new shared artifacts use `/workspace/project/roborender-artifacts`. Historical
path-bound data/releases and active renderer runtimes remain at the paths below.
No folder migration changes scientific contracts or admits a new native run.

Source: `/home/ldu/roborender` on Leo and `/workspace/project/roborender` on shared
storage. Live workers use immutable hash-pinned releases. Matching Git HEAD
alone does not attest uncommitted code. Keep large artifacts and secrets out of Git:

- RTX native bundle: `/home/ldu/BC_render/rtx_BC_transfer`, including Raven source,
  robot/scene/motion assets, manipulation teacher and both Homie checkpoints.
- RTX anchors: `/home/ldu/roborender_assets/anchors/anchor_pool_2048`.
- H200 render assets: `renderer/models`, `renderer/offline_teacher`,
  `renderer/online_image3f`; the SM90 engine is not an RTX engine.
- Student bootstrap/spec: `checkpoints/students/bootstrap/`.
- Shared data and evidence: `data/BC/`, `checkpoints/`, ignored `runtime/`.

Read `pipeline/deployment/LEO_MIGRATION.md` before changing host bindings and
`pipeline/deployment/WORKING_REPO.md` before source synchronization. Leo's owned
Cosmos route does not require Alpha, Beta or a Mac.

## Scientific invariants

**October20 source correction:** new Isaac6 overlays consume already-applied
initial robot-refresh flags at the shared `reset_all()` completion boundary for
BC collection, DAgger training/heldout collection and testing. Ordinary episode
resets are unchanged. New entrypoints reject old overlays; existing data and
frozen runs are not retroactively corrected. Native activation remains gated;
see [reset fix, tests and exposure](INITIAL_RESET_FIX_AND_EXPOSURE_20261020.md).

- Policy views: head/left hand/right hand, vertically tiled, 108×192 each;
  RGB `[0,1]` and ImageNet normalization exactly once.
- Student: ImageNet ResNet18, visual feature 128 + actor state 113, SiLU MLP
  `[512,256,128]`, deterministic 11-dimensional action mean.
- Final action: 11 manipulation outputs + the unchanged 15-D Homie tail.
  Uniform MSE supervises only teacher action dimensions `[0:11]`.
- Teacher labels every visited state; teacher, Homie and renderers are frozen.
- Offline renderer slots are left/right/head; online Image3F slots are
  head/left/right. See `renderer/contracts/CONDITIONING_CONTRACT.md`.
- Preserve one full configured B×81 optimizer/BatchNorm population; no silent
  microbatching or previous-round/window replay.
- Raven source is pinned at `94d79851d9c471b1210be81c75116d9bea835a4d`.
  Three views, held-out stopping, mixed-policy rounds and motion cohorts are
  explicit extensions; see `pipeline/RAVEN_EQUIVALENCE_AUDIT.md`.

## Pre-BC collection and rendering

The completed clean collection plan used **2,560 training chunks** and
**320 originally validation chunks**, all using the corrected reset. New BC
uses their admitted train-only union above; do not recollect or split them anew
merely because validation is disabled.
Do not adopt the old 2,000+560 training pool or old validation data into it.
Every chunk has 81 synchronized frames, actor states and teacher labels.

```python
freeze_assets_seeds_and_whole_chunk_quotas_before_collection()
while teacher_demonstrations_needed:
    controls, state = capture_current_three_view_isaac_state()
    teacher26 = frozen_teacher11(state) + frozen_homie15(state)
    outcome = env.step(teacher26)
    append_pre_action_controls_state_and_label(controls, state, teacher26)
    emit_complete_81_row_within_episode_chunks()
    discard_partial_tail_at_real_episode_end(outcome)

for chunk in admitted_whole_chunks:
    rgb = offline_teacher50(chunk, steps=50, cfg=5, seed=42,
                           max_chain=1, previous_clip_reference=None)
    save_rgb_and_unchanged_synchronized_state_action_rows(rgb, chunk)
```

Training covers all295 motions: 200 contribute nine chunks and95 contribute
eight. Validation covers every motion once plus25 preselected second quotas,
using separately seeded cohorts and preselected positions over each motion's
8–9-complete-chunk training exposure span. Preserve chronological81-row chunks;
do not select by reward/success/loss or split frames from one trajectory across
roles. Record actual reset/episode/source/render lineage and reject leakage.
Matching motion/exposure support is not proof of equal task-stage frequencies.
See `CLEAN_RECOLLECTION_PLAN_20261020.md`; the October15/16 reuse plans are historical.

## BC training and selection

New runs use the no-validation `fixed_final` protocol described above. The
following stopped continuation and original capped recipe are historical.

**Historical October22 user override for the clean-data BC continuation:** resume
the selected epoch16 policy/AdamW/RNG using the unchanged admitted datasets and
full16×81 batch. Early stop requires complete epoch-average training-mode MSE
strictly below0.015 **and** validation patience8 exhausted. No epoch cap applies
to this continuation (`--no-epoch-cap`); runtime/safety errors still stop.
Validate and save each epoch; retain minimum-validation checkpoint separately.
That continuation was subsequently stopped by the user; epoch116 is the
selected baseline (checkpoint SHA `2ec87a3c7b76aaa8fbb57d4fd50d02335dbfed75c164f5d741bea0067a33d7a9`).
**Historical October22 authorization (superseded by October24):** after its corrected fixed295 success test
and W&B receipt finish, run paired B16-per-RTX capacity admission followed by
first-round reliability qualification only (teacher75%, two real cohorts/four
full B32 updates). Use a fresh source/run and `--qualification-only`; preserve
the evaluator until terminal/quiescent, then attest any required source-owner
transition. Stop at `QUALIFICATION_COMPLETE`, keep successor owners warm and
retain H200 compute. This does not authorize later rounds or an unrestricted
production queue. See `DAGGER_SCALING_20261022.md`. Other frozen runs retain their
original budgets. The table below describes the original capped BC run.

| Parameter | Historical original BC setting |
|---|---|
| Train | 2,560 chunks / 207,360 frames |
| Validation | 320 chunks / 25,920 frames |
| Full training batch | 16×81 = 1,296 frames; 160 updates/epoch |
| Optimizer | AdamW, LR 0.0002, weight decay 0, gradient clip 1 |
| Validation cadence | Baseline and every full epoch; entire held-out set |
| Budget | Minimum 10, maximum 100 epochs |
| Plateau patience | 8 complete validation checks |
| Meaningful improvement | Greater than `max(1e-6, .005 * patience_best_mse)` |

```python
student, optimizer = pinned_bootstrap_and_AdamW()
measure_full_validation_baseline()
for epoch in remaining_epochs_up_to_100:
    for batch in sorted_cyclic_full_B16_batches(train_index):
        loss = mean((student(batch.rgb3, batch.actor_obs) - batch.teacher11)**2)
        backward_clip_and_step(loss)
        log_train_mse(loss)
    val = evaluate_entire_validation_set_without_gradients()
    save_checkpoint_with_optimizer_rng_and_stopping_history()
    retain_absolute_lowest_validation_checkpoint(val)
    update_separate_meaningful_improvement_patience(val)
    if epoch >= 10 and patience_count >= 8:
        break
run_corrected_fixed295_success_test(selected_checkpoint)
```

For this clean experiment, initialize from the approved pre-BC bootstrap, not
an old exposed student/optimizer checkpoint. Strict resume applies only within
the newly admitted experiment and dataset identity.

Activation recomputation preserves the full configured B×81 BatchNorm population. Sorted
cyclic loading is the retained Raven-compatible default, not random shuffling.
Strict resume restores model, optimizer, RNG, sample/data identities and stopping
history. Historical validation checkpoints and best aliases remain unchanged;
new fixed-final runs do not create validation metrics or best-validation aliases.

## DAgger training and final-checkpoint handoff

Rounds use teacher probabilities **75%, 50%, 25%, 0%**. Carry the previous stage's
final policy **and AdamW state**: the explicitly selected BC epoch116 for round1,
then the preceding round's update100 for each later round. No DAgger validation
set is collected or consumed under the new fixed-budget protocol.

```python
for beta in (.75, .50, .25, 0):
    initialize_scene_once_and_bind_fixed_lane_motions()
    for window in range(100):
        begin_model_rollout_buffers()  # Raven's model lifecycle, not env.reset
        freeze_one_student_checkpoint_for_all_actors_and_workers()
        for t in range(81):
            capture_controls_state_and_teacher_labels_on_RTX()
            rgb = require_genuine_H200_Image3F_three_view_render()
            student11 = H200_student(rgb, actor_state)
            action26 = durable_beta_selection(student11, teacher26)
            await_all_local_lane_selections()
            outcome = RTX_env_step_once(action26)  # native done/timeout resets only
            idempotently_commit_outcomes_to_H200(outcome)
        clear_model_rollout_buffers()
        require_all_actors_and_all_complete_chunks()
        one_full_batch_MSE_optimizer_update()
        save_register_and_activate_checkpoint_at_barrier()
        if window == 3:
            await_bound_live_prefix_qualification_without_restarting_Isaac()
    require_exactly_100_accepted_updates_and_no_pending_update()
    final = immutable_checkpoint_and_AdamW_at_update_100()
    run_corrected_rendered_fixed295_success_test(final)
    require_complete_295_result_and_online_logging_receipt()
    next_round_parent = final
```

B16+B16 uses one physical scene per actor ×100 continuous81-step windows/round.
Each update is 32 chunks/2,592 transitions; each round is 3,200 chunks/259,200
transitions, with fixed32-motion support and no padded rotation. Each fresh
window is the full learner batch; no old-window replay or within-window update.
The first4 qualification updates are the verified prefix of100, not an extra4.
The same accepted native command/PID/source must wait at that gate and continue
through100. A controller may reattach to proven surviving commands, but a saved
model/AdamW checkpoint cannot restore lost physical/RNG state. Uncertain work
fails closed; never silently reconstruct a continuous round from a checkpoint.
Provider: `pipeline/deployment/continuous_production_collection.py`.
Historical fixed100 plans used10 cohorts ×2 windows ×5 coverage epochs; preserve
their receipts without labeling them continuous-round execution.

### Historical held-out-MSE protocol (not the new launch default)

The **historical B2+B2 baseline** uses 74 four-motion cohorts, each with
two continuous windows: 295 unique motions + one declared padding motion.
One coverage epoch is 148 updates / 592 chunks / 47,952 frames. Its minimum is
2 and maximum 10 coverage epochs; patience 5 full checks and the same .005/1e-6
threshold (October 17 user update). The no-improvement earliest stop is epoch5.
Previously frozen patience-3 runs keep their historical identity; the later
historical MSE-selected plans used patience 5. Neither rule applies to fixed100.
The cap is 1,480 updates / 5,920 chunks / 479,520 frames per round.

The **historical MSE-selected B16+B16 plan** had 10 cohorts per coverage epoch, each with two
continuous windows: 20 updates / 640 chunks / 51,840 frames. Each update is one
full 32x81 batch, not microbatches. All 295 motions are covered; 25 explicitly
declared padded slots rotate across epochs. It validated the frozen320-chunk
heldout set every 20 updates with patience 5 and minimum 2/maximum 10 coverage
epochs. Fixed100 removes that heldout stage; the fixed295 test layout is retained.

These counts must be regenerated and reviewed when production batch size is
increased; do not just replace B2 in a launcher while retaining hard-coded
coverage and learner contracts. Temporal windows are not independent episodes.
An81-row window may cross real episode ends. In the new continuous-round
protocol, its boundary must not reset physical episode/RNG state, while model
rollout buffers follow Raven's per81-step lifecycle. The historical protocol
above replaced scenes only at explicitly admitted cohort boundaries; it is not
the new continuous default.

Historical runs sampled50% ray-traced/50% generated per episode. **New runs
always use generated observations**, independently of the teacher/student
behavior draw. Retain a fixed episode anchor and Image3F5 / CFG1 / TeaCache.3.
Teacher labels every state, including those reached under student behavior.

## Mandatory corrected success test: BC and every DAgger round

The October24 scaled-test override above supersedes the following B2 layout
for the replacement evaluator. Keep this paragraph as the historical contract
for frozen B2 receipts, not permission to import them into scaled295 results.

Use the same exact **295 cases, motion assets, ordering, seeds and cohort layout**
after each stage. The benchmark is 147 B2 cohorts + one B1 cohort; changing the
training batch does not change this benchmark's initialization distribution.

Production students explicitly select `student_clear_already_applied_once_v1`.
After original initial reset writes and observations, clear only the already-
applied pending robot-refresh flags once. This prevents a first-step robot rewind
without matching objects. Ordinary episode resets and success criteria are unchanged.

Evaluate the selected student only: deterministic action mean + Homie, three
Image3F-generated views of each current state, no teacher takeover or optimizer.
Use the versioned rendered fixed295 panel; retain its ray-traced predecessor as
historical comparison evidence only. Native horizon is 8 seconds,
control 50 Hz, physics 200 Hz. OR-latch native goal until first done; final score
is `successes / 295`, only when all cases complete.

The authoritative version and evidence are in
[TEACHER_SUCCESS_EVALUATION.md](TEACHER_SUCCESS_EVALUATION.md) and executable
`pipeline/deployment/rtx_actor/production_success_contract.py`. Production native
admission, result import, logging, handoff and resumed campaign gates reject
old-reset/missing-version results, changed panels, teacher takeover and incomplete
coverage. Corrected command/output identities cannot reuse old cached outcomes.

A fresh frozen source release and worker qualification are required before live
use. The remaining old diagnostic worker is not corrected production deployment.
Native teacher evidence does not claim a new student success result. Same fixed
cases/seeds/protocol is enforced; bitwise PhysX/RGB determinism is not asserted.

## Stopping meaning, limits and fairness

Historical BC and held-out-MSE DAgger stopping sought generalization, not training loss driven to zero.
Patience compares against the last meaningful minimum, so small cumulative gains
can reset it. The lowest validation checkpoint is retained separately. Hitting
`max_epochs` or `maximum_iterations` is a **budget stop, not proven convergence**;
report that distinction. Do not silently extend budgets or tune them using the
success test. The current logic allows testing at a cap without a plateau.

New fixed-final BC and fixed100 DAgger use a **budget stop, not proven convergence**.
There is no validation-MSE claim; log train MSE and final success only.
The last checkpoint is intentionally selected even if an earlier loss was lower.

BC historical provenance limits remain. Historical DAgger's fixed parent-policy validation
is fair for checkpoint comparison within a round but may not match all later
student occupancy; cross-round validation MSEs use different populations. Full
motion support does not guarantee identical task-stage frequencies. Repeatedly
tuning research choices on the fixed success panel turns it into a development
benchmark; use an untouched test for final generalization claims.

## Reliability and logging

Only RTX steps. Idempotent prepare/commit retries never replay physical actions.
H200 alone assembles learner-complete chunks. Do not train on partial windows or
restart uncertain simulator state from only a model checkpoint. Keep Kit warm
across supported scene changes; preserve checkpoint, optimizer and evidence.

Historical four-worker forwarding uses18500–18503 (Image3F),18620–18623
(render/student),18610–18614 (control/evaluation). New eight-worker routing
requires its own exact endpoint/owner admission; never reuse the four-worker
mapping for both RTX actors. Log real per-update train MSE and separate
stage-end `test/*` success metrics; held-out MSE belongs only to explicitly
selected historical validation protocols. Bind logs
with source/panel/protocol/checkpoint identities and online W&B readback.
Full payload rescans do not belong in the per-step hot path.

Use one durable ten-minute watchdog, not duplicate observers/controllers. Check
actual owner/command/receipt/step progress and checkpoint/renderer identity;
notify on phase change, completion or faults, remain quiet if healthy/unchanged.
On faults diagnose immediately and repair/test the failed component. Reattach
only at proven durable boundaries. An SSH timeout is not native death; never
blindly restart/replay Isaac or claim a checkpoint restores physics/RNG. Preserve
and exclude uncertain partial windows; require admitted fresh-lifecycle recovery.

Never overwrite a frozen release or reset unrelated dirty work. Preserve models,
datasets, credentials and runtime history outside Git. Read the standing-Isaac
and round-boundary recovery runbooks before any actual recovery operation.

### Training / success-test phase isolation

New BC/DAgger training has no held-out MSE pass or validation selection. BC uses
offline demonstrations and an explicitly configured finite epoch budget;
DAgger uses fresh online trajectories and exactly100 accepted full32×81 updates.
Do not infer a new BC budget from historical patience/cap settings. Preserve
BC116 and all old frozen protocol identities without relabeling them.

Required transaction (not a claim of deployed integration):

1. **Finish training:** stop accepting new rollout/update requests at the declared
   boundary. Drain accepted work; reconcile request IDs, committed transitions
   and learner receipts. Save final policy, AdamW, RNG/counters and lineage using
   file+directory fsync and atomic publication. Verify its SHA before publishing
   the training-complete receipt. No partially completed update is a final model.
2. **Enter testing:** require both exact native owners IDLE and no in-flight
   stepping/learning. Preserve Kit/AppLauncher and GPU/process identity. Install
   the test assignments, seed/reset profile and phase-specific renderer sessions;
   clear rollout/RNN history, done/goal latches, actions, anchors/prompts and source
   choices. Load the committed checkpoint strictly and freeze it. Compatible
   scene reuse is allowed; changed scenes rebuild safely within the warm Kit.
3. **Test:** independent RTX actors each use their assigned four H200 renderers.
   Enforce the all-lane render/action barrier before each Isaac vector step.
   Student input is generated three-view Image3F, never native RGB fallback.
   There is no teacher takeover or optimizer update. Count native task success
   through first-done, not an action-MSE threshold. Complete each exact panel case
   once, including the real7-case tail; preserve failures and partial-cohort
   uncertainty. Aggregate only matching checkpoint/panel/reset/render receipts
   and verify W&B readback before reporting COMPLETE.
4. **Return to training:** after terminal test and both-owner IDLE receipts,
   install the next training assignments, beta, phase RNG/counters, histories and
   renderer namespace. Load the prior round's final policy+AdamW; never inherit
   test simulator state. Ordinary81-step windows are not phase resets.

BC optimization itself does not step Isaac; both runtimes may remain loaded and
physics-idle. Warm does not mean artificial activity or zero scene-reset cost.
On uncertainty, preserve receipts and fail closed without replay. A dead Kit or
changed source requires an explicitly admitted fresh lifecycle, not a hot patch.
Do not promise survival of external preemption or hardware/process failure.

Existing native guards/tests live in
`pipeline/deployment/rtx_actor/{resident_isaac_trainer,test_resident_phase_transitions}.py`.
The legacy campaign hook uses single-actor evaluation/four renderer workers.
The new retained-site opt-in uses `paired_scaled_runtime` and rejects scaled
panels without its exact paired runtime admission; no B2 fallback. Evaluation capacity is
currently checkpoint-bound; a later checkpoint needs its own admission, not a
relabelled BC116 proof. The latest required bounded qualification is
train→scaled-test→train→scaled-test: four genuine full32×81 updates per training
leg and checkpoint-specific generated B16+16 testing after each, with separately
bound native and learner sources. New curriculum certification uses one full
frozen release for both; historical split-source evidence is not relabeled.
Require unchanged Kit identities, terminal/IDLE
receipts, isolated assignments/sessions/histories, zero test optimizer steps, and
exact policy/AdamW/learner-RNG continuity. At a declared phase boundary, discard
unfinished simulator episodes explicitly and exclude incomplete learner windows;
never count them as completed successes or carry their state into the next phase.
This does not authorize resets at ordinary81-step window boundaries. Final test
completion leaves both owners warm/IDLE and compute retained. Evaluation B16 memory
evidence does not qualify training B16 memory or prove a global maximum batch.

### Training curriculum continuity (2026-10-25)

Carry Raven's restorable training curriculum scalars per RTX actor across rounds
using `dagger_rtx/training_curriculum.py`. Bind each export to the exact completed
training command, source, owner and final checkpoint. Restore after cold-phase
preparation and before the existing initial `reset_all`; observe that reset once.
Never import evaluation curriculum, physics, lane histories or simulator RNG.
The production boundary remains update100; the separately admitted certification
boundary is update4. Missing/mismatched predecessor evidence blocks the handoff.
Two independent B16 curricula are not mathematically the same as one global B32
curriculum; preserve and disclose this scaling distinction. Test clocks/panel
semantics remain unchanged. Implementation/CPU tests are not live certification:
require the new source-bound train→test→train→test receipts before claiming that.

### Separate Raven-style success benchmark

The user explicitly requested a separately versioned Raven-style benchmark.
Do not change fixed295 cases, clocks, receipts or historical scores. The new
opt-in benchmark randomizes initial episode/stage clocks in Raven's order and
pools completed episodes, including every simultaneous completion on the final
vector step. Each RTX is an independent16-lane replicate with a150-episode
minimum target; retain its actual denominator and report both replicates plus
a weighted aggregate. This is not one synchronized global32-lane/150-episode
pool. A32-episode-per-actor qualification is a smoke result, not the full benchmark.
Generated-only pure-student input, corrected initialization/terminal accounting,
frozen weights, zero optimizer updates and clean warm phase isolation remain
mandatory. Preserve explicit motion-support and cold-evaluation-curriculum
differences; do not claim numerical replication of Raven's published score.
