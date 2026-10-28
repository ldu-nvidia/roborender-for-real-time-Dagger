---
name: roborender-distributed-pipeline
description: Operate, audit, debug, or extend the canonical RoboRender BC and distributed RTX-to-H200 DAgger pipeline, including Isaac teacher collection, offline rendering, Image3F, student optimization, checkpoint provenance, and independent closed-loop success testing.
---

# RoboRender distributed pipeline

Work from the canonical Git checkout. Read `README.md`, then “Current decisions
and rationale — authoritative summary” in
`docs/END_TO_END_PIPELINE_CONTRACT.md`, then the owning stage's README.
The contract is the single current protocol authority; this skill is its
operating guide, not a second set of defaults. That summary records the reasons
for the Raven-continuity/BC-replay corrections and explicit experiment exceptions.
Versioned contracts, frozen configs, manifests and actual runtime receipts are
authoritative. Historical logs and old machine profiles are evidence, not defaults.

**October25 certification request:** qualify initial-bootstrap B32 BC, then
rendered B16+16 testing, then continuous B16+16 DAgger and warm phase handoff.
RTX0 uses H2000–3; RTX1 uses H2004–7, with distinct admitted worker/GPU identities.
The earlier B16 BC smoke is not B32 evidence. No premature chunk-count Isaac
reset, native-RGB student fallback, held-out validation, live source patch or
historical relabeling. The active BC lineage and actual certification gates/status
are defined in the contract; documentation/CPU tests alone never certify them.

## Host and process ownership

- The host-bound RTX actor owns persistent Isaac, privileged teacher, Homie,
  episode/RNN/RNG lifecycle, and the only `env.step([B,26])` call. Dedicated
  `leo-rtx-6k` uses physical GPUs0/1; display GPU2 is excluded. Read
  `pipeline/deployment/LEO_MIGRATION.md` before touching assets or GPU bindings.
- The CPU access pod or RTX hosts editing and orchestration. Reserve H200 only
  for concrete training/rendering/inference under approved
  `cosmos-solution-architect-xcwtffiq`, using `custom.gpu.*xh200`. Offline BC
  has qualified one-H200 workloads; the legacy training topology uses four
  H200 workers, while scaled success testing qualifies eight. Training routing
  and warm phase round-trips need separate proof. Do not reserve GPUs for agents.
- H200 training workers own Image3F, student inference and optimization. Scaled
  success testing keeps frozen student/Homie/Isaac on RTX, Image3F on H200, and
  performs no optimization. CPU access
  services own the registry, coordinator/evaluation ledgers and orchestration;
  these control services must not require an idle GPU reservation. **October15
  user override:** retain a secured RoboRender H200 allocation until the user
  explicitly asks to release that allocation. Completion, failure, timeout or
  stopping a workload is not permission to delete/scale down its allocation.
  Separate process cleanup from allocation release; report retained capacity
  and never manufacture GPU activity to defeat cluster reclamation. Cluster
  preemption/administrative removal remains outside this retention guarantee.
  Never stop unknown workloads.
- RTX source lives in `pipeline/dagger/rtx` within this unified repository.
  Models, engines, datasets, caches, credentials and runtime journals stay out
  of Git. Read `pipeline/deployment/WORKING_REPO.md` before synchronizing source.

Leo has approximately62GiB host RAM; free RTX VRAM alone does not qualify a
batch. Require measured full-phase host/cgroup peaks, headroom, swap/OOM counters
and exact environment cardinality. Capture-only, full BC buffers and online
DAgger have different memory footprints. Do not add `MemAvailable` to total
owned cgroup usage as if reclaimable file cache were new capacity.
For new BC campaigns, require the v2 storage policy and reserve/identity guard;
observed compressed chunk sizes are not guaranteed maximum output sizes.

## Scientific invariants

- Policy views are `head_color`, `hand_left`, `hand_right`, vertically tiled at
  108×192 per view. Online Image3F maps ext1=head, ext2=left, wrist=right;
  the original offline teacher renderer maps left/right/head. Do not silently
  unify these distinct original interfaces.
- Student predicts11 manipulation actions; the final26-D behavior action
  appends the exact supplied15-D Homie tail. Teacher labels every state,
  irrespective of whether teacher or student behavior is selected.
- Online controls retain Raven h5_v4 depth and textured robot-on-black RGB at
  240×416, with one fixed three-view anchor per episode. Observation-source
  is **generated-only** for new student training/testing: Image3F5 online and
  teacher50 offline BC. Native RGB/depth may condition rendering but must never
  bypass it into the student. Missing rendering fails closed. Preserve randomized
  initial episode phases; old50/50 runs remain historical.
- An online update collects81 sequential vector transitions for every admitted
  lane, then performs one full B×81 forward/optimizer update. Do not silently
  microbatch the BatchNorm population. An81-row DAgger window can span real
  episode boundaries; native BC chunks are within-episode and discard partial
  terminal tails. An81-step boundary is not an Isaac reset.
- New DAgger uses the continuous-round schema: one physical scene per actor for
  all100 updates, native done/timeout resets only, Raven model buffers begin/
  clear every81 steps. New full-reset-inventory configs retain all295 motions
  on both actors, with sampler lane offsets0/16 and local physics indices.
  Initial32 lanes are not an inventory restriction. First4-update qualification
  pauses the same command without a scene
  restart. Legacy two-window lifecycles require explicit compatibility opt-in.
  A checkpoint cannot restore lost physical/RNG state; do not replay uncertain
  native work. Read the contract/audit before activation or equivalence claims.
- Mixed-policy rounds use teacher probabilities .75/.50/.25/0. No replay across
  iterations or rounds. Carry the explicitly admitted policy **and optimizer
  state** into the next round; never silently substitute BC116 for a newly
  trained parent or discard incompatible optimizer slots. Production fixed100
  uses precisely update100 for the next round; diagnostic checkpoints remain
  separately scoped and cannot be promoted as completed production rounds.
  Teacher probability1 is BC regardless of a run's name.
- New offline rendering uses `raven_3view_50step_independent_v2`:50 steps, CFG5,
  seed42, max_chain1, no inherited references between clips. Keep legacy
  six-clip data for reproducibility; do not relabel it or its old splits.
- BC retains one complete admitted B×81 forward, uniform11-D action MSE, pinned
  AdamW/clipping/normalization and sorted cyclic sample order. New B32 requires
  separate live evidence; do not substitute two B16 normalization populations.
  Preserve Raven's offline81-forward replay lifecycle. Strict resume restores
  optimizer, RNG, counters, dataset/sampling identity and stopping history.

Read `pipeline/RAVEN_EQUIVALENCE_AUDIT.md` before claiming equivalence. The
three-view requirement and mixed-policy rounds are deliberate differences;
held-out stopping and bounded motion-cohort scheduling are historical only.
Source similarity is not
numeric physics equivalence or proof of matching a reported single-view score.

## Train, validation and final test

For historical provenance only, `docs/CLEAN_RECOLLECTION_PLAN_20261020.md`
describes the retired2560+320 collection. The old2880 union and BC116 lineage
are retired; never restore or select them as a fallback. New collection/training
needs explicit user authority, fresh provenance and qualified corrected-reset
source. The user-authorized fresh smoke parent is diagnostic, not production.
The new fixed100 DAgger collects fresh training only, no held-out chunks; all
stages use the explicitly selected versioned success protocol: fixed295 or the
separately admitted Raven-style pooled benchmark. Never mix denominators or
silently replace one with the other. Old reuse and held-out DAgger plans are
historical only.
The user requests removal of old problematic collections: resolve exact paths,
check live consumers, preserve provenance and reusable assets/models/test cases,
then delete only confirmed collection targets; never delete a broad data root.
No documentation edit certifies runtime readiness or releases compute.

New BC uses **explicit epoch budget → final complete epoch → rendered success
test**; no implicit budget or validation fallback. New DAgger uses **100 updates
→ final-checkpoint rendered success test → next round**. Keep data-integrity
checks and existing held-out files; do not log invented validation metrics. Read
`pipeline/dagger/MSE_VALIDATION_PROTOCOL.md` for its explicit opt-in schemas.
Its held-out rules apply only to those historical/explicitly selected schemas.

Before training, freeze the expected motion population, assets, renderer,
train/validation seeds and final-test cases. Reject source/episode/window/hash
or render-reference leakage. Matching support alone is not matching task-stage
frequency; report sampling limitations and native eligibility evidence.
Native episode strings are lane-local, not global IDs. Preserve them verbatim
and use the versioned run/round/actor scope bound to the actual reset receipt.
Do not parse an actor from an episode name, relabel old collection runs, or treat
new run names alone as proof that two reset/trajectory populations are disjoint.

For historical held-out-MSE protocols only, DAgger validation trajectories are separately collected under the round's frozen
parent and beta, then held fixed for the entire round. No validation chunk may
enter the learner. Measure the parent baseline first; preserve the absolute
lowest-MSE checkpoint while a separate meaningful-improvement threshold controls
patience. This is parent-occupancy validation, not a guarantee of matching all
later on-policy states. Maximum budget and validation cadence must be explicit.
W&B and candidate metadata must name the actual opt-in metric protocol; an outer
configuration ceiling is not the execution budget. A pending candidate does not
inherit its BC parent's validation MSE as its own measured score.

Only after immutable checkpoint selection, execute the pinned success panel
with no optimizer. Report actual successes/completed episodes and observation
mode. Success values must not choose checkpoints, adjust patience or trigger
extra tuning against that same test. Small smoke panels certify functionality,
not policy quality. Never promise a target success rate.

## Change, verification and recovery

Inspect Git status and artifact registries before edits. Preserve frozen live
releases; an intended change requires a fresh source/config/run identity or an
explicitly admitted control-only overlay. Verify identical per-file source on
both hosts; matching source manifests are not a Git commit or push.

Run focused CPU tests, compilation, shell/JSON checks, `git diff --check` and
secret/large-file scans. Before a live run require actual asset/checkpoint hashes,
host/GPU identity, genuine endpoint identity, and scoped credentials. A CPU test
or fake conformance result cannot substitute for native evidence.

Native acceptance includes an all-lane prepare barrier before exactly one
vector step, durable commit receipts, synchronized81-row learner artifacts,
real optimizer/checkpoint receipts, protocol-required inference, final success outcomes,
and online W&B readback. Keep evidence under ignored dated runtime directories.

Live SQLite is single-host, including readers. Use owner RPC/pinned SSH, never
open an H200 live ledger from CPU shared storage. New ledgers are owner/boot-bound
DELETE/FULL; old WAL or uncertain ACKs require offline reconciliation and a fresh
namespace. See `docs/LEDGER_DURABILITY.md`; an integrity check is not receipt proof.

Never replay after `CRASH_UNCERTAIN`, step a partial lane batch, or treat an RTX
audit seal as a complete H200 learner chunk. Preserve/exclude incomplete windows.
Warm owners may change scenes only at explicit lifecycle boundaries; ordinary
window boundaries preserve episode/recurrent/RNG state. Read the standing-Isaac
and round-boundary-recovery runbooks before recovery; a lost simulator is not
restored by merely loading a policy checkpoint.

Use one ten-minute watchdog with a real durable scheduler/agent-delivery route.
Check progress and receipts, not just process liveness. On failure diagnose/test
the fix, reconcile ownership/intents/commits and recover only from a proven safe
boundary. Notify on faults/phase changes/completion; quiet when healthy/unchanged.

If round1 fails during held-out collection before learning, use only the
reconciled fresh-namespace procedure in `docs/PRETRAINING_RECOVERY.md`; preserve
completed BC/test and exclude partial data. `docs/NATIVE_MEMORY_POLICY_V3.md`
defines bounded PSI warnings without waiving OOM/swap/headroom checks; source
tests alone do not admit a new native lifetime or relabel old failed receipts.

After a reconciled round1 training failure, the distinct whole-round restart
in `docs/FAILED_FIRST_ROUND_RESTART.md` can preserve completed BC/test only.
It excludes the entire failed round and starts again from BC policy+AdamW in
a fresh namespace. Never use pretraining recovery to bypass existing learner
jobs, or claim that restart admission fixes a native fault.

Leo's authenticated Cosmos route does not require Alpha, Beta or a Mac. Read
`pipeline/deployment/connectivity/README.md` before changing connectivity. Keep
unrelated sessions, credentials, data and GPU jobs untouched.

## Training ↔ testing boundary

Read “Training / success-test phase isolation” in
`docs/END_TO_END_PIPELINE_CONTRACT.md`; runtime guards live in
`pipeline/deployment/rtx_actor/resident_isaac_trainer.py`, tests in
`test/deployment/rtx_actor/test_resident_phase_transitions.py`.
Keep all first-party tests/helpers under repository-root `test/`; see
`test/README.md` and run `bash test/run_cpu_checks.sh`.
Do not move runtime safety guards into tests or rewrite pinned upstream tests.
Use the explicitly selected BC checkpoint or DAgger update100, corrected rendered
fixed295 test, then a fresh training reset—not test state. Commit/fsync the final
checkpoint, drain accepted work, require both owners IDLE, clear histories/actions/
anchors and install phase-specific assignments/RNG/render sessions. Keep both Kit
processes warm and physics-idle between compatible phases. Require frozen inference
and zero optimizer steps in test. Source replacement needs a fresh attested lifecycle;
fail closed on uncertain transitions. Never relabel legacy single-actor hooks as
the paired eight-worker handoff or reuse checkpoint-bound capacity proofs blindly.
CPU tests are not native round-trip qualification; never patch a live frozen release.
