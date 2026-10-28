# October 16 production handoff

**Historical plan:** superseded for the next experiment by
[October20 clean recollection](CLEAN_RECOLLECTION_PLAN_20261020.md).
Do not reuse its old data, queue IDs or launch defaults for the clean run.

This records the then-authorized sequence, not a claim that collection or
full-population training has already finished. Runtime state and immutable
receipts take precedence over this document.

## Data and admission

The completed base contains 2,000 historical teacher trajectories rerendered
with the independent 50-step three-view recipe, plus 320 separately collected
and rendered held-out chunks. Its distinct legacy admission is preserved.
Historical missing motion/reset/seed provenance is not reconstructed or hidden.

The active extension selects 560 new whole 81-row chunks across all 295 motion
files: 265 motions contribute two chunks and 30 contribute one. Complete chunk
ordinals are chosen before collection across a nine-chunk exposure span, not
selected by reward, success or loss. Process seeds are disjoint from recorded
historical, validation and fixed-test seeds. This cannot prove nonduplication
against unknown historical seeds.

Single-collector native sessions use the measured B6/B1 profiles, a 43 GiB
cgroup cap, zero owned swap/OOM/limit pressure, an 8 GiB host floor and bounded
disk admission. Transfer consumes completed-session prefixes. The render
successor waits for the complete native population, then uses four retained
H200 workers: 50 steps, CFG 5, seed 42, `max_chain=1`.

`pipeline/bc_training/training_extension.py` creates a new **2,560 train /
320 validation** admission only after reconstructing collection, execution,
capacity, transfer and genuine rendering evidence. It checks source/render
identities and known cross-role provenance, without repeatedly decompressing
all image payloads. Deterministic deep samples supplement complete metadata and
small paired-state/action checks. The old admission is not overwritten.

## BC production

The CPU queue is under `runtime/experiments/20261016-prod-1/bc/`. It waits for
the actual 560-render completion and expanded admission before starting one
optimizer process on GPU0 of the existing four-H200 allocation.

| Parameter | Value |
|---|---|
| W&B | `nvidia/dagger`, `2026-10-16-prod-1-BC` |
| Initialization | Immutable three-view ImageNet bootstrap; not an old student |
| Full forward population | 16 chunks × 81 rows; 160 updates/epoch |
| Loss | Uniform mean squared error on all 11 manipulation actions |
| Optimizer | AdamW, learning rate 0.0002, gradient clipping 1.0 |
| Ordering / seed | Raven-compatible sorted cyclic ordering / 42 |
| Validation | All 320 held-out chunks, baseline and every epoch |
| Stopping | Minimum 10, maximum 100 epochs; patience 8 |
| Meaningful improvement | `max(1e-6, 0.005 * patience_best_mse)` |
| Selection | Absolute minimum complete validation MSE, independently of patience |
| Checkpoints | Every validation; model, AdamW, RNG and sampling/stopping state |

Training logs MSE per update and validation MSE per epoch. After immutable
selection, a separate native RTX test executes the frozen 295 cases. It does no
optimization. Success results cannot select a checkpoint or tune this run.
An evaluation request or registered case list is not a completed success test.

## DAgger production

The qualified online memory baseline is **two Leo RTX owners, B2 each**, with
four H200 render/student workers. Spare VRAM alone does not authorize B6+B6:
that historical profile observed host/cgroup limit pressure. A future larger
profile needs its own complete-buffer and native continuity qualification.

Each window is 81 sequential vector transitions, one synchronized label/action
pair per state, followed by one full **4×81** learner update. The 11-D student
action gets the exact supplied 15-D Homie tail. Teacher labels are always
collected. Episode-boundary resets are native; a window boundary never resets
Isaac, recurrent history or RNG.

The full-motion scaling extension explicitly changes scenes at declared cohort
lifecycle boundaries, not every window: each four-motion cohort runs **two
continuous windows**. Seventy-four cohorts cover 295 unique motions plus one
declared padding motion. Thus **148 updates form one coverage epoch**. New seeds
and namespaces prevent ambiguous service episode identity across cohorts;
actual native reset samples and local service episode identity remain attested.

Four rounds use teacher probabilities **75%, 50%, 25%, 0%**. Each carries the
previous MSE-selected model and AdamW state, without previous-round replay.
Names are `2026-10-16-prod-1-dagger-teacher75`, then `teacher50`, `teacher25`
and `teacher0`.

Before each round, collect a separate fixed 320-chunk held-out population across
all 295 motions using that round's frozen parent and behavior mixture. It is
never submitted to the learner. Evaluate complete held-out MSE at baseline and
after each full coverage epoch. Minimum two / maximum ten coverage epochs,
patience five checks (October 17 user update for new plans), relative improvement 0.005 and absolute improvement
1e-6 govern stopping. Select the absolute-lowest validation-MSE checkpoint,
then execute the independent frozen 295-case success test.

The first two genuine cohorts are a provisional runtime qualification: four
updates, 1,296 combined transitions, both physical GPUs, full-batch optimizer
and checkpoint continuity, reset/assignment receipts, zero unsafe memory pressure and
actual resource samples. Only a completed verified prefix can continue. CPU
unit tests do not replace this native gate; an uncertain active phase cannot
be replayed or relabeled as completed production.

## Retention, observability and recovery

Retain `leo-rr-leo-e4838f7a5c31-r1`, owned by
`dagger-leo-bc-reuse2000-20261015`, on `node-ip-10-50-96-203` under
`cosmos-solution-architect-xcwtffiq`. No workflow here authorizes allocation
release, cancellation, scale-down or teardown. Ending a workload/observer is
not allocation release. External administrator/platform preemption remains
possible; do not manufacture GPU utilization to prevent it.

Use immutable per-phase source releases, actual collection/render/learner
receipts, route-scoped idempotency and exact checkpoint barriers. Keep native
Kit warm between supported phases. Log real substep timings, counts, errors,
MSE and success outcomes, never invented progress or placeholder metrics.
Completed cohort/round boundaries may be resumed only through their receipts;
persisted service files alone cannot restore simulator/RNN/RNG state after an
uncertain crash. Preserve all evidence and checkpoints for reconciliation.

The collection and BC queue have their own bounded CPU supervisors. Supervisor
deadlines detach from GPU work and retain compute; they are not resource expiry
permissions. A launch readiness claim must state which actual native and
cloud-logging gates have passed. No success-rate target, including 70%, is
guaranteed by infrastructure qualification.

### Native memory and test continuation correction (October 17)

The standing B2 resource policy v2 keeps each worker's exact **21 GiB hard
cap**, `memory.swap.max=0` and an **8 GiB host reserve**. OOM, owned swap,
changed process/cgroup identity and unknown counters remain fatal. It no longer
confuses Linux's cumulative `memory.events.max` cache-reclaim counter with an
OOM: reclaim is admitted only with at least 1 GiB measured working-set margin
and bounded memory PSI (`some avg10 <= 10%`, `full avg10 <= 5%`). Actual raw
max counters, safe-cache classification, unsafe events and phase deltas are
preserved separately in versioned receipts. They are never replaced with zero.
This is an infrastructure correction, not a change to actions, rewards,
physics, renderer inputs, goal accounting, sampling seeds or stopping rules.

Every complete independent success-test cohort has a durable receipt bound to
the original full panel, checkpoint, case IDs and native outcomes. A fresh
native owner may reuse only a verified completed prefix; incomplete episodes
are excluded, not resumed from service metadata. The final denominator remains
295 unique completed cases. Partial counts are not the BC success rate.

The retained H200 pod also supports an explicitly pinned private SSH route
from the CPU access pod and a CPU-owned, loopback-only relay for Leo. These use
the existing H200 host key and allocation/replica/four-device identity checks,
without the old interactive cloud TTY/cat helper. Reconnection replaces only
the relay's own SSH child, never the native owners, H200 daemon or allocation.
Legacy route contracts remain unchanged unless this route is explicitly bound.
