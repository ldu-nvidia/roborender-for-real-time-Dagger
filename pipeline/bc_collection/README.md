# BC collection

This stage runs **Isaac teacher demonstration collection on Leo RTX only**.
It saves simulation conditioning, observations and action labels. It does not
run the photorealistic renderer, train the BC student, or calculate success rate.
Those are separate downstream pipelines. H200 is not needed for this stage.

**Current work, not yet production-certified:** the user requests a validated
paired32+32 smoke followed by one RTX×32 production collection with a minimum
10,000 complete chunks and natural final-episode draining. `episode_drain.py`
contains the new stop-accounting state machine; plan/ledger/writer integration
has CPU differential/integration tests. A bounded functional probe accepts a
small total target and at most16 windows; it cannot produce a capacity receipt
or authorize production. It is not enabled in the current
fixed-window smoke and does not bypass production/capacity/storage gates.
The explicit `--production-collection --target-total-chunks 10000` plan stages
only the one-RTX B32 protocol. Execution requires separately rebuilt, hash-bound
current-source full-buffer capacity and actual natural-drain proofs on the same
GPU/recipe. The small drain run must finish recorded episodes and exclude new
episode rows; zero-age stopping is not proof. Production uses rolling disk
admission (81GiB free per B32 window,64GiB reserve, actual archive-size bounds),
without claiming whole-campaign fit. Space shortage or safety-ceiling exhaustion
is failure, never successful quota completion. Legacy whole-campaign budgeting
is unchanged. These draft CPU-tested changes are not live certification.
Stopping after natural closure may exceed10,000; short terminal tails remain
discarded. The total-target drain is an explicit extension to Raven's finite
window budget. See the current contract and runtime HANDOFF for actual status.

The canonical editable folder on Leo is
`/data1/workspaces/roborender/source/pipeline/bc_collection`.
Run from a pinned release of that source, not a mutable checkout or a home-path
copy. The [current contract](../../docs/END_TO_END_PIPELINE_CONTRACT.md) governs
new runs; the [Raven comparison](../RAVEN_EQUIVALENCE_AUDIT.md) records evidence
and the limits of equivalence claims.

## Raven procedure and our replication

Reference: Raven commit `94d79851d9c471b1210be81c75116d9bea835a4d`, particularly
`gr00t/rl/config/exp/loco_manip/wsdpt_videogen_collect.yaml`, the PPO/distillation
rollout lifecycle, and `distill_trainer_videogen.py` methods `policy_step`,
`_rollout_step` and `_save_collected_chunks`. These are pinned dependencies,
not another editable production implementation in this folder.

| Collection rule | Raven reference | This implementation |
| --- | --- | --- |
| Parallel population | Collection YAML defaults to 32 vector environments | One shared 32-lane scene by default; not two independent 16-lane scenes |
| Motion assignment | Lane-index mapping into the loader's ordered motion inventory | Freeze the full eligible order; first 32 entries assigned to lanes, with the full inventory retained for native sampling |
| Repeated sampling | Native resets sample new initial conditions within each lane's assigned reference motion | Keep the scene and curriculum alive; repeat on natural done/timeout, not by restarting short collection jobs |
| Initial clocks | Randomized episode/stage clocks after the initial reset | Preserve that initialization once; do not re-randomize all clocks at every window |
| Behavior and labels | Privileged manipulation teacher plus Homie locomotion | Execute the same finite 26-D teacher/Homie action that is recorded as the label; no student behavior |
| Window | 81 sequential vector steps | Same 81-step boundary for capture/save and model-context bookkeeping, not a physical reset |
| Saved data | Within-episode 81-frame chunks | Append each row, emit when length reaches 81, then discard any remaining terminal tail |
| Nonterminal tail | Survives a collection-window boundary | Carry it into the next window; never join rows from different episodes |
| Selection | Save complete emitted chunks, not only successful episodes | No success filter, per-motion quota, or first-8/9-chunks selection |
| Run length | Finite configured outer-loop budget | Explicit `max_windows`; repeats/chunk counts are outcomes, not per-lane targets |

An **environment** is a live simulator lane. A **reference motion** supplies that
lane's task/motion condition. An **episode** is one sampled attempt until native
termination. Repeated attempts under the same motion can differ because reset
conditions and randomization differ. An **81-step window** is a vector collection
unit; an **81-row saved chunk** is an episode-local data unit. Their boundaries
need not coincide after lanes terminate at different times.

There is no fixed "repeat N times per environment" in this procedure. Faster
episodes may repeat more often. For budget W and B lanes, the collector executes
`81 × W` vector steps (`B × 81 × W` lane transitions), but saves fewer rows when
short terminal/final tails are discarded. It does **not** promise every complete
episode is stored in its entirety: short tails are intentionally omitted, initial
clocks are randomized, and the overall budget can end during an episode.

### Collection pseudocode

```text
freeze full eligible motion order, seed, B=32 and explicit window budget W
construct one Isaac scene, privileged teacher, Homie and camera hooks
reset the scene once; randomize initial episode/stage clocks as Raven does
keep an episode-local pending buffer for each lane

repeat W times:
    begin model rollout context and an empty 81-step capture window
    repeat 81 times:
        capture pre-action observations, depth and robot RGB for each view
        action = teacher manipulation (11) + Homie locomotion (15)
        step all B lanes once using that action; record labels and done/timeout
        let native done/timeout reset individual lanes; other lanes continue
    end model rollout context (do not reset Isaac)
    convert depth using the original whole-window normalization
    for each lane, in temporal order through this window:
        append row to its episode-local buffer
        if buffer reaches 81 rows: emit chunk and clear those emitted rows
        if row is terminal: discard remaining tail and break episode chaining
    retain nonterminal partial buffers for the next window
    preserve Raven's incidental learner RNG draws, without optimizing

finish at the declared budget; do not pad or flush incomplete final tails
verify terminal receipts and publish all complete emissions as a raw pool
```

The teacher is the **learned privileged upper-body policy**, not an alternative
to a privileged controller: it supplies 11 manipulation actions from privileged
state. Homie supplies the other 15 locomotion actions. The future BC student
learns the manipulation component; no student policy controls collection.

### Explicit differences and evidence limits

We retain corrected robot/object initialization, the Isaac 6 compatibility port,
and the required head/left/right conditioning views. The supplied three-view
Raven collection configuration already supports those views; single-view results
are not interchangeable with this experiment.

Raven's collection runs through its general trainer. Here the collection loop
is standalone: incidental student optimization on native images is omitted,
consistent with generated-only student training. The RNG bridge preserves the
discarded learner random draws; this is not a claim of bitwise physics identity.
Hashes, provenance, storage/owner guards and train-only publication are explicit
operational additions. See the qualification section below: source/CPU parity
does not certify full-resolution B32 capacity or a complete production campaign.

## Runtime interface and canonical paths

`run_continuous.py` is the collection entrypoint; its reusable runtime boundary
is `pipeline/deployment/rtx_actor/bc_runtime.py`. New runs use the pinned
`configs/storage_paths.json` and save an immutable resolved-path snapshot.
Assets are under `/data1/workspaces/roborender/simulation_assets`, caches under
`/data1/workspaces/roborender/caches/isaac`, and outputs under
`/data1/workspaces/roborender/runs/bc-continuous/<run-id>`. No home/data2 aliases
are selected by the new BC launcher. Collection/writer algorithms are unchanged.
See [runtime ownership and control](../deployment/rtx_actor/BC_RUNTIME.md).
Docker's system-managed image store remains an explicitly separate admin task.

## Standalone collection implementation (October26)

New plans default to `roborender.bc-standalone-collector.v1`. The stage owns:

- `collector.py`: one initial reset and randomized clocks, then repeated81-step
  vector windows in the same scene. Calls the teacher/Homie/camera runtime hooks
  and native `env.step`; never invokes the general PPO `train()` loop.
- `chunk_writer.py`: extracted pinned Raven depth conversion and NPZ writer,
  including append→emit81→discard terminal tail and cross-window carryover.
- `continuous_plan.py`: default32 environments, one scene, complete ordered
  motion inventory, explicit overall window budget, all emissions selected.
- `run_continuous.py`: source-pinned RTX launcher and unchanged resource gates.
- `continuous_ingest.py`: complete-run provenance and dataset publication.
- `storage_guard.py`: filesystem identity and free-space reserve checks; no legacy controller dependency.
- `episode_drain.py`: draft versioned total-chunk target/recorded-episode drain
  accounting; no Isaac steps, resets, ownership management or resume mechanism.
- `streaming_storage.py`: separate disk policy for the drain probe. Before each
  window it requires64GiB reserve plus512MiB per possible emission and1GiB
  metadata (81GiB forB32). Each actual NPZ is bounded from uncompressed array
  sizes before writing, then verified. This does not guarantee the whole
  campaign fits. Disk exhaustion is failure, never successful completion.

The shared Isaac adapter still constructs the native environment/policies and
attests resets/actions/files. It is a runtime bridge, not the collection-loop
owner. Native teacher/camera hooks remain pinned Raven dependencies; Isaac assets
are not duplicated here. The new loop performs no optimizer, validation,
scheduler, checkpoint-saving callback or learner-batch preparation. Raven's
discarded learner RNG draws are retained through the validated RNG bridge.

Natural terminations repeatedly restart individual lanes while other lanes
continue. There is no equal repeat count per lane, no8/9-chunk cap, no success
filter and no81/162-step physical reset. Only the explicit overall window budget
ends normal collection; unfinished final episodes/tails are not forced complete.
The adapter accepts Raven's inclusive reference-sample interval endpoints.

Historical requests without the new protocol retain their explicit compatibility
path. Do not silently relabel their evidence. New completion/ingestion records
bind both stage-module hashes; the launcher mounts the frozen collection package
read-only. CPU differential tests live under `test/pipeline/`, including real
original-loop comparisons and a32-lane repeated-episode writer test past nine
chunks. These are not live Isaac/B32 memory qualification. No runtime is started
by this change, and the bounded-storage production gate remains closed.

## Current protocol: continuous Raven collection (October25)

### Explicit paired32 diagnostic (October26 Pacific)

The user-authorized test harness is `test/pipeline/run_paired_bc_smoke.py`.
It launches fresh32-lane scenes on physical GPUs0/1 for four81-step windows
each, seeds0/1, using the bounded writer. Both keep the first32 motion conditions
from the full pinned inventory. This tests64 live lanes, not64 distinct motions.
`paired_bc_guard.py` permits simultaneous capture/stepping and serializes heavy
CPU saves; the peer waits without resetting or losing its episode buffers.
The4GiB host reserve is opt-in for this profile only. Single-collector8GiB
defaults,43GiB actor caps, GPU floors, and zero owned swap/OOM remain unchanged.
An unavailable memory read holds native progress and permits bounded read-only
rechecks; it is not reported as OOM. Persistent unknown state, changed identity
or proven unsafe counters stop only the exact owned pair and exclude partial
work; no replay or compute release. Diagnostic results cannot be
published as a production pool or used as the single-owner capacity certificate.
The paired live test remains unqualified until terminal native/payload/resource
evidence passes. This introduces no disk-spilled episode tails.

Attempt `bc-pair32-20261027-050100` did **not** pass: the observer rejected an
actor sample after GPU0/GPU1 accepted2/3windows. The original observer omitted
the rejecting sample, leaving its exact cause unknown. Sampled host memory
available never fell below11.626GiB; this is not proof of OOM or proof of safe
capacity. The logging bug is fixed in editable source and regression-tested;
the frozen failed attempt, partial outputs and all compute remain preserved.

October27 observer repair: `bc_resource_observer.py` binds the exact Docker ID,
image, PID start time, boot ID and cgroup inode once, then samples host cgroup
files directly (no per-memory-sample Docker RPC). Actual kernel memory/no-swap
limits must match the launch. Every attempt/rejection is journaled before abort.
Unknown reads set `observation-hold.json`; paired collectors wait at safe
step/lane boundaries without resetting or resubmitting work. Three fresh reads
are allowed for a bound owner; initial acceptance has up to20 attempts within
a12-second retry budget. Native holds fail closed after30seconds if observation
does not recover. Real pressure/identity violations never receive a retry waiver.
205 focused regression tests and a real CPU-only Docker fault-injection probe
passed. The probe recovered the same PID after one injected read failure, with
actual `memory.swap.max=0`. This is observer validation, not a new B32x2 capacity
pass: a fresh immutable paired collection qualification is still required.

### Explicit 40/50-lane chunk-target diagnostic (October26)

The user-requested smoke uses one RTX,40 or50 lanes and
`--target-chunks-per-lane 20 --max-windows 40` with the GPU-window writer.
This binds `roborender.bc-all-lanes-chunk-target-smoke.v1`, not a new default.
Save each complete native chunk immediately and stop at the first full-window
barrier where every lane has at least20 saved chunks. Faster lanes continue;
retain their extra chunks. Never pause/reset lanes at their target, truncate
episodes, pad tails or count simulation windows as saved chunks. Natural
done/timeout resets remain unchanged. The40-window ceiling and runtime timeout
fail incomplete if the target is not reached; neither is successful completion.

Launch remains `--qualification --execute`, but this diagnostic cannot create a
capacity certificate or publish a production dataset. Existing4/8-window capacity
qualification and ordinary B1..40/default32 plans are unchanged. Keep43GiB host
cap,8GiB host floor, device reserve, no swap/OOM and detached owner-local logging.
This is an experimental B50 test, not established safe capacity. Outputs remain
under `runs/bc-continuous/<fresh-run>/bc_isaac`, never inside version-controlled
source. A fresh source-bound admission is required; no failed run is replayed.

After the B50 attempt stopped at72 vector steps on the retained GPU-window
headroom guard (zero completed chunks), the user requested a fresh B40 run.
The same target, stop semantics and memory limits apply; neither width is yet
certified stable. The failed B50 run and source remain unchanged.

The first B40 attempt also stopped: host cgroup usage reached41.08GiB of43GiB
during its first save, with13 uncommitted NPZ files and no accepted window.
The user authorized repair and a fresh B40 target smoke. The bounded writer
now scopes conversion arrays to one lane helper, compacts unused host heap at
lane barriers and records before/after memory in `writer-memory.jsonl`.
Episode tails, raw GPU snapshots, numeric conversion and NPZ order stay intact.
This is a candidate memory-lifetime repair, not a proven root cause or capacity
certificate until the fresh live run finishes. No guard, cap or reserve changes.
That first repair completed two windows/54 committed chunks but stopped before
the third save on the projected reserve. A further repair fsyncs each completed
owned NPZ and advises away only its file cache, after writing and again after
indexing. Records are in `output-cache-events.jsonl`; advice does not guarantee
eviction, so guards still read actual memory. No global cache drop or data
deletion. The full40-lane target remains unverified until a terminal PASS.

**October27 result — bounded B40 target smoke PASS:**
`/data1/workspaces/roborender/runs/bc-continuous/smoke40-cachefix-20261026-231200`
completed27 windows/2187 vector steps and854 complete chunks,20–23 on each of
40 lanes. One initial reset_all, natural subsequent resets, zero optimizer steps.
All payload/provenance hashes and81-frame array headers were independently
verified after normal completion. GPU peak70.813%; cgroup peak35.688/43GiB;
zero limit/swap/OOM events.151 focused CPU tests passed. The repair uses
per-lane scratch lifetimes, unused-heap compaction and fsync-before-file-cache
advice, without changing chunk bytes, episode/reset/RNG logic or safeguards.
Frozen source SHA256:
`6e8f069b4cd821706f25e726cc23b8fd49a6d41373d2523de3db16f4a3c1b370`.
Evidence: `runs/bc-smoke40-cachefix-20261026-231200/verified-result.json` plus
the run's `continuous-execution.json`, `host-result.json` and native manifest.
This proves this one-RTX bounded smoke, not a maximum batch or unrestricted
production campaign. The diagnostic remains `production_admitted:false`;
it cannot publish a production dataset or replace the separate capacity gate.

Use `continuous_plan.py` → `run_continuous.py` → `continuous_ingest.py` for new
BC collection. Plans and dry runs perform no GPU work; an executed run requires
a fresh source-pinned namespace, ownership, storage/memory guards and genuine
full-buffer capacity evidence. No old dataset is restored or selected.

- One continuous vector scene, reference default32 lanes, preserves the shared
  native termination curriculum. Two separate16-lane scenes are not equivalent.
- Freeze the resolved full eligible motion order. Keep all files in the native
  reset/resampling loader; active lanes use its firstB entries without rotation.
  The current profile rejects fewer eligible motions than lanes rather than
  silently changing the mapping. A supplied historical directory order is not
  inferred from today's sorted inventory.
- Seed once (default0), retain native randomized initial episode/stage clocks,
  and reset only on native done/timeout. No physical reset at81/162 rows or after
  a per-motion quota. An explicit vector-window budget controls run length.
- Keep **all** complete native81-row chunks, including their actual terminal
  provenance. Preserve append→emit→done-tail order, short-tail discard,
  nonterminal cross-window buffers and depth normalization before reassembly.
  No success filter, padding, forced episode completion or final partial flush.
- Publish a train-only pool. There is no held-out collection, MSE-validation
  partition or ordinal/chunk quota. Metadata and hashes supplement the unchanged
  native NPZ payloads; they do not relabel partial work as complete.

The supplied Raven collection YAML already has head/left/right conditioning
cameras. Keep corrected robot/object initialization and the Isaac6 compatibility
port. Collection performs zero optimizer updates: Raven's incidental learner
work on native images is intentionally omitted under the generated-input-only
training rule. Accordingly, no bitwise RNG/physics-equivalence claim is made.

CPU/source tests are not live certification. The new full-buffer collection
profile requires its own current-source native qualification; scaled success
testing and the old B6 quota collector do not qualify continuous B32 collection.
Do not stop or hotpatch warm owners, revive retired data, or release H200 compute
to run this profile. Any lost simulator lifetime is incomplete, not resumable
from a chunk index or student checkpoint.

### Planning, qualification and publication

`python -m pipeline.bc_collection.continuous_plan --help` documents the staging
interface. Supply either an explicitly ordered JSON list of
`{motion_id,path,sha256}` descriptors or `--motion-dir` plus its container-visible
`--native-motion-root`. Directory resolution uses Raven's `os.listdir` order,
not lexical/hash order; native construction checks eligibility and all hashes.
The plan requires `--max-windows`; its unit is81 vector steps, **not episodes or
saved chunks**. Default `--num-envs 32` describes one logical scene. A longer
budget increases exposure without restarting motions/curriculum or selecting
particular chunk ordinals.

`python -m pipeline.bc_collection.run_continuous --help` documents the guarded
launcher. Without `--execute` it verifies source/request metadata and prints
the budget only. `--qualification --execute` is a separately authorized4/8-window
native probe; it cannot publish production training data. Production execution
requires the actual current-source capacity receipt and its hash. It refuses
existing warm Isaac owners rather than stopping them; any required owner
replacement needs its own explicit admission. Storage, host RAM, cgroup, GPU,
swap/OOM and exact-container ownership remain guarded throughout execution.

`python -m pipeline.bc_collection.continuous_ingest --help` documents terminal
publication. It checks the actual launch/host/native receipts, complete window
chain, all NPZ/provenance hashes and within-episode row chronology. It publishes
a fresh flat `isaac/*.npz` pool plus streaming JSONL metadata, usable as input to
offline rendering. It never marks raw conditions as generated training images,
sets active dataset selectors or adopts a partial/qualification run. Old
quota/split ingesters must not be used to reinterpret this new schema.

## End-to-end runtime procedure on Leo

These are operator steps, not evidence that a workload has run. Perform the
source/CPU checks first; do not append `--execute` until the corresponding native
qualification or production launch is authorized and admitted.

### 1. Prepare source, assets and an exclusive runtime

Start at `/data1/workspaces/roborender/source`. Review the central
`configs/storage_paths.json`, then freeze source with the repository's
`pipeline.deployment.source_snapshot` tooling. The launcher must execute from
the exact immutable `code_releases/<release>/code` or `frozen` directory named by
its verified manifest. Editing the canonical checkout does not update an already
frozen release or an active container.

| Component | Host location / container role |
| --- | --- |
| Canonical Python source | `/data1/workspaces/roborender/source` |
| Raven code, motions, robot/object assets, teacher/Homie checkpoints | `/data1/workspaces/roborender/simulation_assets/rtx_BC_transfer` |
| Anchor assets required by the shared container setup | `/data1/workspaces/roborender/simulation_assets/anchor_pool_2048` |
| Rebuildable Isaac caches | `/data1/workspaces/roborender/caches/isaac/leo-gpu0` or `leo-gpu1` |
| Fresh run directory | `/data1/workspaces/roborender/runs/bc-continuous/<run-id>` |
| Isaac installation | `/isaac-sim` inside the pinned Docker image, not a host Python environment |
| Per-run compatibility overlay | `<run-dir>/overlay`, exposed inside the container as `/isaac6-port` |
| Raw native output | `<run-dir>/bc_isaac`, exposed as `/dagger-runtime/bc_isaac` |
| Docker image storage | System-managed `/var/lib/docker`; not yet migrated to the project disk |

The immutable image ID is selected by
`pipeline/deployment/rtx_actor/qualify_native_isaac6.py`; do not substitute a
floating tag. Source/assets are mounted read-only; the run directory and caches
are writable. Source changes normally require a fresh pinned release, not an
Isaac image rebuild. The full container/mount builder is shared runtime code,
not duplicated in the scientific collector.

Use physical RTX GPU **0 or 1**, never display GPU2. This BC profile is one
32-environment scene on one admitted GPU. It is not the separate B16+16
DAgger/evaluation topology. Check actual GPU ownership, host/cgroup RAM, disk
reserve and active Isaac containers. The launcher takes an exclusive host lock
and refuses existing native owners; it does not stop them to make room.

### 2. Stage the collection plan and request (no simulation)

Use a fresh staging directory separate from the fresh run directory. In the
following template, `SOURCE` is the verified frozen source root, `STAGE` is the
operator's staging directory, `RUN_ID` is a fresh identifier, `WINDOWS` is the
explicit approved budget, and `MOTIONS_JSON` contains the ordered descriptors.
Each descriptor contains `motion_id`, the **container-visible** motion `path`,
and its SHA256. Do not invent or silently sort the historical motion order.

```bash
cd "$SOURCE"
python3 -m pipeline.bc_collection.continuous_plan \
  --eligible-motions "$MOTIONS_JSON" \
  --run-id "$RUN_ID" --num-envs 32 --process-seed 0 \
  --max-windows "$WINDOWS" \
  --output-dir /dagger-runtime/bc_isaac \
  --plan-out "$STAGE/plan.json" --request-out "$STAGE/request.json"
sha256sum "$STAGE/request.json"
```

Alternatively use `--motion-dir` and `--native-motion-root` as described above.
Native startup independently checks the full inventory, eligibility and bytes.
The default host-buffer storage path and the explicitly selected bounded-memory
qualification profile below are distinct; do not switch profiles after staging.

### 3. Verify the launcher, then qualify capacity before production

`SOURCE_MANIFEST` names the matching source manifest; `REQUEST_SHA256` is the
actual raw request-file hash printed above. `RUN_DIR` must be a nonexistent
direct child of the canonical `runs/bc-continuous` directory. `STORAGE_ROOT`
identifies its admitted physical filesystem; the guard checks filesystem
identity, budget and reserve rather than trusting a free-space number alone.

```bash
cd "$SOURCE"
python3 -m pipeline.bc_collection.run_continuous \
  --request "$STAGE/request.json" --request-sha256 "$REQUEST_SHA256" \
  --source-root "$SOURCE" --source-manifest "$SOURCE_MANIFEST" \
  --runtime "$RUN_DIR" --storage-root "$STORAGE_ROOT" \
  --physical-gpu 0 --num-envs 32 --memory-gib 43 \
  --timeout "$TIMEOUT_SECONDS"
```

This command is a **dry run**, not capacity proof. A separately authorized
4/8-window probe uses `--qualification --execute` and its own fresh namespace.
Production requires a new request/run and `--execute --capacity-receipt PATH
--capacity-receipt-sha256 SHA256`, with genuine matching current-source evidence.
The bounded GPU-window writer remains qualification-only, even after a probe;
the original full-B32 host-buffer path has an unresolved capacity gate. Do not
weaken limits, borrow evaluator receipts, or treat this template as a currently
certified B32 production launch.

### 4. What an admitted execution does

1. Verify source/request hashes, host/GPU identity, exclusive ownership and
   storage/memory admission; create the fresh run and freeze resolved paths.
2. Materialize and hash the per-run Isaac 6 compatibility overlay from the pinned
   Raven bundle. Write the launch intent before starting the exact container.
3. Start the pinned image with the selected GPU, memory controls, read-only
   source/assets and scoped writable mounts. The native adapter constructs
   Isaac, the scene, teacher, Homie and camera hooks and verifies initial state.
4. Call `collector.collect`: execute the procedure above in one simulator
   lifetime, repeatedly saving full episode-local chunks through `chunk_writer`.
   Shared adapter hooks enforce memory/storage guards and write provenance.
5. At the declared window budget, validate native completion and resource
   evidence and write the terminal execution receipt. The standalone collector
   is a finite container workload, not the warm train/test resident service.
   Cleanup only targets its identity-verified owned container.

Monitor the scoped run's `intent.json`, `launch.json`, `container.cid`,
`memory-samples.jsonl`, `host-result.json`, `continuous-execution.json` and
`bc_isaac/cohort-manifest.json`, plus its native logs. Existing chunk files or a
zero process exit code alone do not establish an admitted complete collection.
For an authorized stop, verify the exact launcher PID and use its SIGTERM/Ctrl-C
handler; never stop arbitrary containers or use Docker prune. Interrupted work
is incomplete: a chunk index/checkpoint cannot restore lost Isaac/RNG state.
Do not replay an uncertain launch. Stopping a workload never releases compute.

### 5. Seal and publish raw demonstrations

Only a successfully completed, admissible production run can be published by
`continuous_ingest.py`. Its CLI requires the original plan, native output,
terminal execution receipt, expected source/adapter hashes and model-asset
identities; obtain these from verified bindings, not guessed values:

```bash
python3 -m pipeline.bc_collection.continuous_ingest --help
```

Publication writes a fresh pool with `isaac/*.npz`, `chunks.jsonl`,
`manifest.json`, `source-seal.json` and supporting evidence. It checks chronology,
all emissions, hashes, terminal/resource receipts and the teacher/model lineage.
Qualification or partial runs cannot be promoted to production data.

The NPZ files contain view conditioning, nonvisual observations and teacher/Homie
labels—not photorealistic student inputs. Original payload `dones` are written
as zeros by the Raven writer; actual episode/terminal provenance is recorded
separately and must not be inferred from that field alone.

Collection ends at the sealed **raw** pool. `pipeline/bc_rendering` separately
generates student images on H200; `pipeline/bc_training` separately optimizes the
student. The raw seal explicitly says `rendered_chunks=0` and
`training_admitted=false`. At this documentation revision, the continuous raw
pool is readable by rendering, but render activation and the default training
selector still need the new-format production handoff; the retired union must
not be restored as a fallback. This README does not certify that downstream gap
as resolved.

### Separate bounded-memory qualification profile (October26)

The optional `--collection-storage-protocol
roborender.bc-gpu-window-lane-writer.v1` on `continuous_plan` selects a new
**qualification-only** implementation, not a new production default. It retains
all four original81-row raw depth/RGB streams on the actor's admitted GPU, then
converts one logical lane at a time on CPU. One sharedB32 simulator/curriculum
population still advances; conversion batching does not partition environments.
Each lane's depth normalization uses its complete original vector window before
episode reassembly. Saved tails own their rows instead of pinning full-window
arrays. RNG, teacher actions, natural reset/emit ordering and final-tail discard
remain unchanged.

This addresses the host-buffer profile's structural memory problem: its B32
raw histories require26.99GiB before native/runtime and conversion memory, while
the old43GiB-cap save guard permits only7GiB current usage. The old guard and
default path are not weakened. The new profile separately reserves the complete
remaining GPU window plus one lane's depth stack, maintains the8GiB/15% GPU
floor, the8GiB host floor,2GiB cgroup margin and zero swap/OOM policy.
Copied tails are accounted separately from the8GiB CPU conversion/NPZ scratch
estimate: B32 retained tails alone can approach10GiB. The save admission computes
their high-water mark from the accepted window's actual lane-major done rows,
including append-before-emit. These payload estimates are not proven allocator
bounds; live full-window capacity remains mandatory.

The profile requires the corrected incidental-RNG protocol, an explicit4/8-window
qualification and distinct writer/source receipts. It truthfully reports
`native_payload_writer_unchanged=false`. Production launch/ingestion remains
blocked without the separate current-source proofs above. CPU tests compare actual pinned Raven writer bytes, chronology, tails,
view-key order and RNG; they do not certify CUDA transfer or native capacity.
Do not apply this source to a running/frozen owner or reuse old capacity proof.

**Memory diagnostics correction (October26):** the host observer now flushes and
fsyncs every sampled observation, including rejecting counters and the specific
failure, before exact-owner cleanup. Native memory-gate rejection also records
its exact snapshot. Swap, OOM allocation, OOM kill and memory-limit events have
distinct host error messages; the old combined error did not identify its cause.
Only this bounded qualification profile may continue after a cumulative
`memory.events.max` event when current container headroom is at least2GiB and
the peak stays within the unchanged40/43GiB cap. The event remains a recorded
warning; native capture/save projections and host/device floors still apply.
This is diagnostic continuation, not capacity acceptance: certification still
requires zero limit/swap/OOM events and all complete-window evidence. Legacy and
production policies stay strict. No automatic retry, replay or limit increase.

## Folder scope after cleanup

This folder contains the implementation modules listed above and this
README. No quota, held-out-validation, cohort campaign or diagnostic entrypoint
remains here. `collector.py` and `chunk_writer.py` are unchanged by this cleanup.

Eight still-imported historical modules now live in
[`pipeline/legacy/bc_collection`](../legacy/bc_collection/README.md). Their callers
were updated explicitly; no legacy forwarding stubs remain in this folder.
The two unused distribution/window audit tools and their dedicated tests were
deleted. Required storage guards were extracted unchanged into `storage_guard.py`.

Pinned Raven materialization/contract checks moved to
[`test/reference/bc_collection`](../../test/reference/bc_collection/README.md).
Regression tests remain under repository-root `test/`, outside production code.
Generated `__pycache__` is not source.

This cleanup does not qualify native B32 capacity, launch workloads, alter frozen
releases, restore retired datasets, or release compute. Source archives retain
recoverability for the removed files.
