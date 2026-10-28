# Historical RoboRender pipeline contract — not production instructions

This superseded document is retained for provenance only. Its old host profiles,
256-case evaluations, six-clip rendering, fixed iteration budgets and resource
release instructions must not be used for new production. Read
[the current contract](END_TO_END_PIPELINE_CONTRACT.md) instead. Cosmos compute
must never be released without the user's explicit authorization.

Contract revision: **2**. Host binding: **2026-10-13**. Qualification update:
**2026-10-14 Pacific** (evidence identifiers use 20261015 UTC).

## October 16 production handoff

The current authorized data/production sequence is documented in
[the October 16 plan](PRODUCTION_PLAN_20261016.md): retain the admitted 2,000
training/320 validation base, finish the preselected 560-chunk training
extension, render it independently, then publish a distinct 2,560/320 admission
before BC. Mixed-policy production adds explicit full-295 motion coverage,
held-out MSE stopping and independent 295-case success testing. These are
declared sampling/scaling extensions; source tests do not substitute for the
new native multi-cohort execution gate. Older plans below remain historical.

## October 15 Raven fidelity and statistical protocol override

**Allocation-retention override, October15:** once a RoboRender H200 allocation
is secured, retain it until the user explicitly requests release. Workload
completion, failure, timeout and process cleanup must not implicitly delete or
scale down the allocation. This supersedes older automatic-release guidance;
it does not authorize unrelated allocations or fabricated GPU activity to avoid
cluster reclamation. Report retained capacity and any externally imposed loss.

**Later October 15 user decision:** execute the explicit
[legacy-reuse protocol](BC_LEGACY_REUSE_20261015.md): all 2,000 historical raw
chunks are training-only, with 320 fresh held-out chunks and a separate
295-case native final test. Preserve the all-fresh 2,880-chunk plan below as a
different experiment. This decision does not restore missing historical
provenance, certify old chained RGB, or weaken the fresh-native admission gate.

This section takes precedence over historical readiness and data claims below.
The clean source reference is commit
`94d79851d9c471b1210be81c75116d9bea835a4d`, available at
`runtime/raven_gr00t_collection_94d7985`. The user's required three-view policy
is retained. Reported single-view success numbers are not a directly matched
three-view baseline. A run named DAgger with teacher probability 1 is BC; names
do not establish which behavior policy collected its trajectories.

### Algorithm invariants and explicit extensions

- Native teacher/action construction, Homie tail, simulator reset/physics,
  synchronized camera/state/action capture, and the original 81-row collector
  are retained. BC chunks remain within episodes and discard incomplete tails;
  online DAgger windows may span real episode boundaries with per-step IDs.
- The supplied three-view rerender launcher uses `max_chain=1`. New offline
  rendering defaults to recipe `raven_3view_50step_independent_v2`, contract
  `raven_three_view_independent_v2`: 50 steps, CFG 5, seed 42, no inherited
  reference from another clip. Generic Raven utilities also support chaining;
  the six-clip legacy pool is not silently relabeled as the supplied launcher.
- BC optimization keeps a complete 16×81 forward population, uniform MSE on
  the 11 manipulation actions, AdamW and the pinned policy/bootstrap contract.
  Sorted cyclic source order is the faithful default; any shuffle experiment
  must be explicitly named and pinned in the resume contract.
- Train/validation/test splitting and validation-MSE patience are requested
  statistical extensions, not protocols supplied by Raven. Train updates use
  only train data. Fixed held-out validation MSE selects the absolute-lowest
  checkpoint; a separate meaningful-improvement threshold controls patience.
  Final closed-loop success data cannot select hyperparameters or checkpoints.
- Teacher probabilities .75/.50/.25/0 are the requested four mixed-policy
  rounds. Each round must declare its behavior mixture, fixed validation
  population, maximum budget, stopping rule and checkpoint handoff explicitly.
  Do not claim that an old fixed-50/success-selected controller has automatically
  adopted the new MSE protocol.

### Data audit and new-data admission

The historical 2,000-chunk six-clip render pool contains 101 render-reference
chains spanning old splits (303 dependent chunks). Its motion/reset provenance
is incomplete. Preserve it for reproducibility, but do not admit it as the new
faithful production dataset merely because all files exist or a resume works.

New collection records motion-file identity/hash, process seed, native reset
and episode identities, stream/rank/lane, teacher-only attestation and a
collection-time split assignment. Scene geometry and the reset sampler must
use the same explicit motion ordering: native filesystem order alone differs
across hosts. Complete preassigned splits must never be redistributed after
rendering. Metadata admission rejects cross-split source, stream, episode,
motion/seed and render-lineage overlap without decompressing all payloads.

The current **unqualified v3 plan**, not a completed production dataset, targets
2,880 chunks: 2,560 train and320held-out validation. Both roles cover295motion
files with independent seed/episode cohorts and preselected matched-exposure
complete-chunk ordinals. A separate295-case equal-motion closed-loop success
panel supplies the final test; unused offline-test RGB is deliberately omitted.
The older3,200-chunk/320-case plan is preserved separately, not silently renamed.
This tests in-distribution generalization, not unseen-motion generalization;
native reset/geometry, true full-buffer capacity and execution admission remain
required. Qualification-only pools cannot be promoted by relabeling them.
The canonical inventory and draft plan are under
`runtime/production-audits/20261015-raven-faithful/`.

### Runtime qualification boundary

The latest full MSE-selected four-round smoke completed B2+B2 native collection,
real H200 optimization, selected model/AdamW handoffs, fixed held-out MSE and
actual two-case final tests with online W&B readback. Both persistent native
owners initialized once; owned swap/OOM/limit-pressure counters stayed zero.
The earlier B6+B6 smoke is functional evidence, not the current full-buffer
memory gate: a historical nonzero cgroup limit-pressure counter prevents
promoting it as a comfortable-capacity measurement. B8 short capture-only probes
also do not qualify complete BC buffers. New guarded B6/B1 BC probes remain
separate from online DAgger. See the dated MSE qualification report.

Leo has approximately62GiB host RAM; a larger batch is not admitted by VRAM
inspection. Never add total cgroup memory.current to MemAvailable, which can
double-count reclaimable cache. Require measured private memory, complete
owned cgroup counters, zero swap/OOM/pressure and a live8GiB headroom floor.
No full-population production claim is valid until actual new data, all native
capacity/geometry gates and long-lived multi-cohort DAgger continuity pass.
H200 allocations are requested only for concrete queued GPU workloads in the
approved reservation. Under the later user retention override above, stopping
those workloads does not release the allocation without explicit user approval.

## October 13 Leo deployment override

The active RTX destination is now the dedicated **`leo-rtx-6k`**, with physical
RTX PRO6000 GPUs **0 and 1**; physical GPU2 is the excluded A400 display device.
Its source is `/home/ldu/roborender`, with immutable live source releases under
`runtime/source-releases`. RTX assets retain the bundle/anchor paths documented
below. CPU orchestration and H200 assets remain on the existing shared volume.
Leo's own authenticated Cosmos connection no longer needs Alpha, Beta, or the
Mac in the runtime path. Read
[`LEO_MIGRATION.md`](../pipeline/deployment/LEO_MIGRATION.md) and its pinned
[`leo_2rtx_migration.v1.json`](../pipeline/deployment/profiles/leo_2rtx_migration.v1.json)
before using the historical Beta GPU indices in the remainder of this document.

Current Leo evidence: both RTX devices passed native B2 capture; fresh B1
teacher-only BC collection completed 243 steps and two 81-frame chunks.
The warm four-round smoke subsequently completed B6+B6 collection, real H200
rendering/inference/learning, checkpoint handoffs, ten-case success evaluation
after every round, and online W&B readback. It reused the completed BC smoke
with explicit provenance. See [qualification](../pipeline/deployment/QUALIFICATION_20261015.md).
The following Beta results remain historical evidence. The tested Leo profile
is B6+B6 with four H200s for DAgger; offline BC uses one H200. Smoke budgets
are one BC update, three updates at each of teacher75/50/25/0, and a separate
complete ten-case success panel after BC and each round. Production remains
50 updates and 256 cases per round. No smoke checkpoint is promoted.

The detailed algorithm below is unchanged by the host migration. A fresh
source manifest and explicit runtime receipts, not this prose or a prior host's
PASS, determine whether each stage has completed.

## Algorithm contract and historical Beta baseline

This is the onboarding contract for teacher demonstration collection, offline
rendering, behavioral cloning, distributed DAgger, and success evaluation. It
defines model inputs, asset ownership, GPU placement, state/action pairing,
and the exact points at which student weights change.

**The corrected BC dataset is ready on H200 shared storage. The selected beta
DAgger deployment has passed three persistent collection windows and three
optimizer updates, not unattended production or full success-rate certification.** The
pseudocode below describes the algorithm; it is not permission to bypass the
remaining launch gates.

Quick navigation: [assets](#model-and-asset-inventory),
[teacher collection](#pre-bc-teacher-collection),
[offline rendering and data](#pre-bc-offline-rendering-and-dataset),
[BC training](#offline-bc-optimization),
[DAgger](#online-dagger-collection-and-learning),
[success evaluation](#success-evaluation-versus-offline-validation),
and [recovery rules](#cross-host-boundary-and-recovery-rules).

## Start here

An incoming agent should:

1. Read this contract and the [current deployment runbook](../pipeline/deployment/WORKING_REPO.md).
2. Read the [beta profile](../pipeline/deployment/profiles/beta_2rtx_4h200.v1.json)
   and its hash-bound [tested candidate](../pipeline/deployment/throughput/candidate_beta_2rtx_4h200.json).
3. Inspect `git status` and the actual run/asset manifests before changing or
   launching anything. Preserve unrelated work and sealed runtime evidence.
4. Use the CPU-only planner to distinguish a runnable BC plan from a blocked
   production/evaluation plan. These commands do not allocate GPUs:

   ```bash
   python3 pipeline/deployment/plan.py --stage bc
   python3 pipeline/deployment/plan.py --stage qualification
   python3 pipeline/deployment/plan.py --stage dagger
   python3 pipeline/deployment/plan.py --stage evaluation
   ```

   The last two intentionally exit 2 while production gates remain open.
5. Do not reuse an abandoned run ID, infer simulator recovery from a model
   checkpoint, or claim success evaluation from training MSE.

The machine-readable profile and a specific run's immutable configuration
bind executable parameters. This document binds their meaning and the
scientific invariants. If source, a run manifest, and this contract disagree,
reconcile them explicitly; do not silently select a historical configuration.

## Current deployment and qualification

| Component | Current placement and responsibility |
|---|---|
| Source and orchestration | CPU access pod, `/workspace/project/roborender` |
| RTX source checkout | Beta host `465d6b0-lcedt`, `/home/ldu/roborender` |
| RTX actor A | Physical GPU1, six environments, one persistent Isaac process |
| RTX actor B | Physical GPU2, six environments, one persistent Isaac process |
| RTX GPU0 | Forbidden for this deployment |
| Online renderer and inference | Four shared H200 Image3F/student workers |
| DAgger learner | One of those four H200 GPUs, after both actors finish the window |
| Offline BC learner | One H200; no RTX, Isaac, or renderer required |
| Offline video rendering | H200 workers; independent complete rendering chains can be distributed |
| Control services | CPU-side orchestration, trajectory metadata, registry, job and evaluation ledgers |

The selected online allocation is `custom.gpu.4xh200` in reservation
`cosmos-solution-architect-xcwtffiq`, with no unreserved/burst fallback. Use
`custom.gpu.1xh200` for the single-device BC trainer. Do not allocate GPUs to
host an agent or keep idle allocations alive. Release owned resources after
completion or abandonment, after checking ownership and workload state.

The currently tested beta actors each have 12 CPU quota and a 21-GiB host-memory
limit, with container swap disabled and opt-in CPU heap compaction at setup
and sealed-window boundaries (limited glibc arenas, no per-step trimming).
Host available memory must stay above 6 GiB and owned containers must not swap.
The October 14 runtime attributes host-wide swap changes using fresh cgroup
counters; a change outside those capped containers is diagnostic, while
missing attribution, owned swap or an OOM still stops the workload.
The earlier throughput trials used a 128-MiB global swap-growth limit.
These are bounded-test settings, not a
claim that RTX VRAM is exhausted or that 12 environments is a global optimum.

| Gate | Status as of this revision |
|---|---|
| Existing corrected BC data and epoch-41 resume | Available; functional resume test passed |
| Beta teacher-only BC collection and H200 teacher rendering | Bounded functional tests passed |
| Two B6 actors, four H200 workers, 81-step windows and full-batch updates | Passed: three windows, 2,916 real transitions, 36 chunks, three optimizer updates |
| Persistent multi-window operation and recovery | Passed for three windows; checkpoint/AdamW continuity and lost-ack recovery verified |
| Four-worker production controller and multi-actor integration | Pending |
| Live online W&B on the beta deployment | Actual training MSE/global-step/update-count cloud readback passed |
| Beta success evaluation and versioned reset protocol | Full paired v4 B5 panel complete: BC 0/256, three-update candidate 11/256; all 52 starts matched, no accounting discrepancies; not legacy B64/Isaac5 equivalence |
| Isaac 6 versus alpha simulator distribution comparison | Pending |
| Beta management independent of alpha and the Mac bridge | Pending |

The earlier October 9 failed multi-window test produced no learner update and must not be
resumed, sealed, or learned from. Its local evidence is
`runtime/runtime-check-20261009/RESULT.md`. It is not overwritten by the later
PASS. Current exact identities/results are in
[`QUALIFICATION_20261012.md`](../pipeline/deployment/QUALIFICATION_20261012.md).

The historical alpha deployment used B28, eight H200 workers and a v4 wire
path. **Do not use its production launcher as the beta launcher or weaken its
guards to force a four-worker launch.** Beta's qualified path is bounded v3
with the separately versioned Isaac 6 migration runtime.

## Model and asset inventory

Persistent assets live on host/shared storage and are loaded onto GPUs only
when needed. Code, configs and manifests belong in Git; weights, engines,
datasets, containers, caches, journals and secrets do not.

### RTX working assets

The beta bundle root is `/home/ldu/BC_render/rtx_BC_transfer`.

| Asset relative to the bundle root | Function |
|---|---|
| `pipeline/GR00T-VisualSim2Real/` | Materialized Raven environment/task source and simulation assets |
| `pipeline/GR00T-VisualSim2Real/gr00t/rl/data/` | Robot, scene/object and motion-reference assets |
| `checkpoints/teachers/robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt` | Privileged state to 11-dimensional manipulation action |
| `checkpoints/teachers/locomotion/g1_locomotion_walk_controller.pt` | Frozen walking controller |
| `checkpoints/teachers/locomotion/g1_locomotion_stand_controller.pt` | Frozen standing controller |

The native source reference is Raven commit
`94d79851d9c471b1210be81c75116d9bea835a4d`. The beta Isaac 6 overlay is a
separately versioned port, not evidence of bit-identical alpha simulation.
Teacher and both Homie identities must be pinned by the deployment asset
manifest; none is optimized during BC or DAgger.

The existing 2,048-anchor pool is
`/home/ldu/roborender_assets/anchors/anchor_pool_2048`. An anchor is generated
three-view appearance data with its prompt, not an action-policy checkpoint.
Use its frame-zero triplet, hold it fixed within an online episode, and
resample on a real episode reset.

### H200 rendering assets

All paths below are relative to `/workspace/project/roborender/renderer`.

| Asset | Role and settings |
|---|---|
| `models/wan21_fun_control_1p3b_foundation/` | Shared Wan2.1 Fun-Control 1.3B foundation and encoders/VAE |
| `offline_teacher/checkpoints/roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors` | Offline BC video and anchor teacher; 50 steps, CFG 5.0, seed 42, tiled VAE |
| `online_image3f/checkpoints/roborender_online_dagger_image3f_5step_lora_step05000.safetensors` | Online Image3F, original `rgb030-depth030-step-5000`; 5 steps, CFG 1.0, seed 42, TeaCache 0.3, untiled VAE |
| `online_image3f/engines/sm90/roborender_online_dagger_image3f_sm90_bf16_3view.engine` | H200 SM90 engine fused from the online adapter; not the offline teacher and not an RTX SM120 engine |

Renderer hashes, preprocessing and inference settings are bound in the
[conditioning contract](../renderer/contracts/CONDITIONING_CONTRACT.md).
CUDA graphs are not implicitly enabled. The historical LADD five-step video
renderer is not required by the current 50-step offline rendering path.

### Student identity

The student is `VisionActorObjPred`, not a GR00T foundation transformer:

```text
three RGB tiles -> vertical stack -> ResNet-18 -> 128-dimensional visual feature
                                                   +
                                           113-dimensional actor state
                                                   |
                                        MLP 512 -> 256 -> 128
                                                   |
                                          11-dimensional action
```

The bootstrap contains the pretrained visual initialization. The visual
backbone and action network train; the object-prediction head and noise
standard-deviation parameters are frozen. No object-prediction loss is added.

Bootstrap and specification:

```text
/workspace/project/roborender/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt
/workspace/project/roborender/checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json
```

The corrected BC checkpoint pinned by the beta profile is:

```text
/workspace/project/roborender/checkpoints/students/BC/bc_3view_autoregressive_v1_20260914T044707Z/best_validation.pt
epoch: 41
global_step: 4100
SHA256: 053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718
```

This is not the older `bc_3view_production_20260907` checkpoint. Resolve the
profile's exact identity; never choose a checkpoint merely because its name
contains `best_validation.pt`.

## Shared observations and actions

| Quantity | Contract |
|---|---|
| Student view order | `head_color`, `hand_left`, `hand_right` |
| Student camera tile | Height 108, width 192, RGB |
| Student image | Vertical stack, height 324, width 192, RGB |
| Normalization | Convert uint8 to float32 `[0,1]` when necessary, then ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]` exactly once |
| Actor observation | 113 float32 values; native actor-state contract |
| Teacher observation | 226 values; privileged teacher input, not student input |
| Homie observation | 516 values with native controller/history handling |
| Student/robot-teacher manipulation output | 11 values |
| Homie locomotion tail | 15 values |
| Outer behavior and teacher-label action | 26 values: `concat(manipulation_11, homie_15)` |
| Underlying environment command | Native wrapper maps into the 31-dimensional robot command; these are not 31 student outputs |

Student behavior is the deterministic action mean. The student does not
directly receive text prompts, depth, masks, privileged teacher observations
or critic observations. Text/depth/robot RGB condition the renderer.

**Isaac supplies observations and transitions, not a ground-truth action.**
The frozen robot policy supplies the 11-dimensional teacher label. Both
teacher and student behavior use the same Homie tail at a given state.

Renderer view slots differ by stage:

| Slot | Offline 50-step teacher | Online Image3F |
|---|---|---|
| `ext1` | `hand_left` | `head_color` |
| `ext2` | `hand_right` | `hand_left` |
| `wrist` | `head_color` | `hand_right` |

Per-view controls are Raven `h5_v4` depth and textured robot RGB on black,
at height 240 by width 416. Do not replace the robot RGB with a binary mask,
swap renderer slots, or feed an arbitrary simulator image as a photorealistic
reference. An absent reference is `None`, not a black image.

## Pre BC teacher collection

Placement: RTX Isaac/teacher/Homie. No student optimization or diffusion
rendering occurs here. Existing migrated data may be reused; collection is
only necessary when a new dataset is requested.

```python
# Conceptual native collection algorithm, not a beta production command.
for actor in independent_RTX_actors_in_parallel:
    env = initialize_native_isaac(actor.seed, actor.num_envs)
    enable_native_random_initial_episode_lengths()
    teacher, homie = load_pinned_frozen_controllers()
    episode_buffers = allocate_one_buffer_per_environment()

    while more_teacher_demonstrations_are_required:
        obs_t = env.current_observations()
        controls_t = capture_synchronized_three_view_controls()
        teacher11 = teacher(obs_t.teacher_obs)
        homie15 = homie(obs_t.homie_obs, native_controller_history)
        teacher26 = concatenate(teacher11, homie15)

        outcome = env.step(teacher26)

        for lane in env.lanes:
            append_to_episode_buffer(
                lane, controls_t[lane], obs_t[lane], teacher26[lane]
            )
            if buffer_length(lane) == 81:
                save_npz_with_original_state_action_and_condition_arrays(lane)
                clear_emitted_frames(lane)
            if outcome.done[lane]:
                discard_incomplete_tail(lane)
                advance_sequence_to_break_cross_episode_chaining(lane)
```

An emitted BC chunk contains 81 consecutive samples from one episode; one
episode can produce multiple chunks. Native collection emits a full buffer
before processing that step's `done`, then clears any incomplete tail. Its BC
`dones` array is zero-filled and sequence gaps break episode chains: do not
treat that array as a complete terminal-event log or a chunk as a full episode.

Saved arrays include `gt_actions[81,26]`, `obs_actor_obs[81,113]`, other
registered non-visual observations, camera controls and source identifiers.
Raw/quantized depth is retained in source data for provenance; it need not be
sent on every DAgger request. Observation at `s_t` is paired with the teacher
action computed at `s_t`, never the observation after executing that action.

The native simulator configuration uses 200-Hz physics and control decimation
four. Simulated control time, video playback FPS and wall-clock throughput
are different quantities; changing an MP4 playback rate does not authorize
dropping or resampling training rows.

Source owners:
[Raven collection contract](../pipeline/bc_collection/raven_bc_collection_contract.v1.json),
[beta qualification wrapper](../pipeline/deployment/rtx_actor/isaac6_bc_trainer.py),
and the materialized `distill_trainer_videogen.py::_save_collected_chunks`.
The current beta BC wrapper is a bounded three-window test, not a bulk
production collector.

## Pre BC offline rendering and dataset

Placement: H200, foundation plus the 50-step step05294 LoRA. All renderer
weights remain frozen. Schedule complete chains across workers; do not render
each chunk independently when it belongs to an autoregressive chain.

```python
chains = group_contiguous_chunks_by_source_stream(max_chain_chunks=6)

for chain in chains_in_parallel_across_H200_workers:
    reference_triplet = None
    for chunk in chain_in_time_order:
        video = frozen_teacher50(
            depth_video=chunk.depth_controls,
            robot_video=chunk.textured_robot_rgb,
            renderer_view_order=["hand_left", "hand_right", "head_color"],
            reference_image_views=reference_triplet,
            prompt=pinned_Raven_render_prompt,
            frames=81, per_view_height=240, width=416,
            steps=50, cfg=5.0, seed=42, tiled_vae=True,
        )
        reference_triplet = full_resolution_last_frame_of_each_view(video)
        policy_rgb = map_views_and_resize(video, height=108, width=192)
        save_rendered_npz(
            original_arrays_unchanged=chunk,
            generated_rgb=policy_rgb,
            source_and_render_chain_provenance=chain,
        )
```

The first chain reference is absent, not Isaac RGB and not an unrelated
online anchor. Later contiguous chunks inherit the previous generated
full-resolution last frames. This reference policy is distinct from the
fixed-episode anchor policy used online. Chains stop at sequence gaps and
their configured six-chunk limit.

Generated arrays are `rgb_gen__head_color`, `rgb_gen__hand_left` and
`rgb_gen__hand_right`; `rgb_gen` remains a head-view compatibility alias.
Downsampling uses the implemented bilinear resize. State/action rows remain
unaltered and synchronized. Training reads NPZ arrays, not encoded MP4s.

Code: [planner](../pipeline/bc_rendering/render_bc_teacher_8gpu.py) and
[worker](../pipeline/bc_rendering/render_bc_teacher_worker.py). The planner's
historical filename does not require allocating eight GPUs.

### Existing shared dataset

Root: `/workspace/project/roborender/data/BC`.

| Relative path | Contents and status |
|---|---|
| `archived/pre_autoregressive_v1_20260913T235716Z/isaac/` | 2,000 original condition/state/action chunks still referenced by the corrected manifest |
| `rendered/` | 2,000 corrected three-view rendered NPZ chunks, ready as BC input |
| `isaac/` | Empty at this revision; not the location of the existing 2,000 inputs |
| `manifests/autoregressive_v1.json` | Canonical source/output mapping and render-chain provenance |
| `splits_autoregressive_v1/train.json` | 1,600 chunks |
| `splits_autoregressive_v1/validation.json` | 200 chunks |
| `splits_autoregressive_v1/test.json` | 200 chunks |

The three split files bind source-manifest SHA256
`67f2652b1d1a8ade14dd2e2a33d98032f3e6bf63ab8f6e2c62a63e19f8fa68ef`.
Splits are grouped by collection stream; do not randomly split adjacent
frames or leak a stream between train and validation. Original conditions
being stored under `archived/` does not make them invalid: the manifest
explicitly reuses them. Do not mix superseded rendered outputs into this pool.

BC does not need alpha, beta, recollection, or rerendering to consume this
existing dataset. Data readiness is separate from GPU allocation, Python
runtime preflight and online W&B authentication.

## Offline BC optimization

Placement: one H200. Load only the learner fields from the stored chunks.
No Isaac/Vulkan initialization, image generation or cross-host actor RPC is
part of an offline training update.

```python
student = load_three_view_bootstrap()
optimizer = AdamW(student.trainable_parameters(), lr=2e-4, weight_decay=0)

if resume_checkpoint:
    restore_model_optimizer_RNG_epoch_step_and_validation_history()
    verify_policy_spec_split_identity_and_batching_contract()
    reproduce_saved_validation_MSE_before_training()

for epoch in remaining_epochs(max_epochs=100):
    student.train()
    for chunks in shuffled_batches(train_manifest, chunks_per_batch=16):
        rgb = normalize_once(vertical_stack_3views(chunks.generated_RGB))
        state = chunks.obs_actor_obs                 # [16, 81, 113]
        target11 = chunks.gt_actions[..., :11]       # [16, 81, 11]
        optimizer.zero_grad()
        loss = mean_square(student(rgb, state) - target11)
        loss.backward()
        clip_global_gradient_norm(student, 1.0)
        optimizer.step()
        log_WandB("train/mse_step", loss, global_optimizer_step)

    student.eval()
    validation_mse = full_held_out_MSE(no_grad=True)
    save_epoch_checkpoint_model_AdamW_RNG_and_provenance()
    update_last_and_meaningfully_best_validation_checkpoints()
    log_WandB("validation/mse", validation_mse, epoch)
    if minimum_epochs_reached_and_validation_plateau:
        break
```

There are 100 optimizer updates per full epoch: 1,600 / 16. Each update uses
16 x 81 = 1,296 observations and uniform MSE across all 11 output dimensions.
The full batch preserves the implemented BatchNorm population. Activation
recomputation saves memory; it is not four independent small-batch forwards.

Validation runs before training/resume and after each epoch, in inference
mode, over all 200 held-out chunks. The current defaults are minimum 10
epochs, maximum 100, patience eight, and meaningful improvement greater than
`max(1e-6, 0.005 * best_validation_mse)`. The best file follows that improvement
rule, not every infinitesimal numerical reduction. Validation batch size is
four chunks and does not change training batch statistics.

`--resume-checkpoint` restores AdamW moments, model, epoch/global step and
available saved RNG state with split/spec/batch consistency checks; it is not
weights-only initialization. See [trainer](../pipeline/bc_training/train_bc_h200.py)
and [launcher](../pipeline/bc_training/train_raven_bc_h200.sh).
The W&B project is `dagger`; the requested BC run name is `BC`.

## Online DAgger collection and learning

### Independent behavior and observation choices

| Round | Teacher behavior probability beta | Student behavior probability |
|---|---:|---:|
| D1 | 0.75 | 0.25 |
| D2 | 0.50 | 0.50 |
| D3 | 0.25 | 0.75 |
| D4 | 0.00 | 1.00 |

Separately, the collection setup assigns a visual source once per episode:
ray-traced with probability 0.5, otherwise generated controls. These are
stochastic probabilities, not exact quotas. Native ray-traced observations
retain Raven brightness/contrast/saturation/hue/blur/noise augmentation.
Generated episodes use Image3F with the fixed episode anchor and its prompt.
Both branches reach the same student normalization and view order.

The teacher labels every visited state. The student is inferred for every
observation, even when teacher behavior is selected. Image3F use depends on
the observation source, not the teacher/student action-selection draw.

### One window and one optimizer update

The following is the selected current-window algorithm. Continuous execution
of all 50 iterations still requires the beta production integration gates.

```python
student, optimizer = load_corrected_BC_model_and_AdamW_state()

for round_id, beta in [(1, .75), (2, .50), (3, .25), (4, 0.0)]:
    initialize_round_with_frozen_run_config_and_new_declared_seeds()

    for iteration in range(1, 51):
        rollout_checkpoint = immutable_current_student_checkpoint()
        require_all_H200_workers_to_use_that_checkpoint_for_this_window()

        parallel_for actor in [RTX_GPU1_B6, RTX_GPU2_B6]:
            for t in range(81):
                # RTX: only this owner may step its six native environments.
                obs = actor.isaac.current_observations()
                teacher11 = frozen_teacher(obs.teacher_obs)
                homie15 = frozen_homie(obs.homie_obs, native_history)
                teacher26 = concat(teacher11, homie15)

                for lane in actor.lanes:
                    if lane.needs_new_episode_binding:
                        lane.anchor, lane.prompt, lane.source = native_episode_assignment()
                    register_required_anchors_once_in_target_worker_caches()

                # H200: sharded across the shared four-worker pool.
                for lane in actor.lanes_in_parallel:
                    label26 = teacher26[lane]
                    if lane.source == "generated_controls":
                        rgb = frozen_Image3F(
                            lane.current_depth_h5v4, lane.current_robot_RGB,
                            lane.fixed_episode_anchor, lane.matching_prompt,
                            steps=5, cfg=1.0, seed=42, teacache=0.3,
                        )
                        rgb = resize_to_policy_tiles(rgb)
                    else:
                        rgb = lane.native_augmented_raytraced_policy_tiles

                    vision = normalize_once(vertical_stack_3views(rgb))
                    student11 = rollout_student(vision, obs.actor_obs[lane])
                    student26 = concat(student11, label26[11:])
                    draw = persist_one_uniform_draw_for_this_step()
                    action26 = label26 if draw < beta else student26
                    persist_prepared_RGB_state_teacher_label_and_selection()
                    return_action_and_matching_receipts_to_RTX(action26)

                # RTX: all six responses must be valid before one vector step.
                await_all_six_prepare_responses()
                outcome = actor.isaac.step(selected_actions_6_by_26)

                # H200 records outcomes; it never asks for another env.step.
                commit_matching_reward_done_timeout_and_identity(outcome)
                bind_new_episodes_only_for_true_native_terminations()

            compact_RTX_audit_seal_and_actor_completion_barrier()

        # H200 is the sole assembler of learner-complete chunks.
        await_both_actors_and_all_12_complete_81_row_chunks()
        batch = load_current_window_RGB_actor_state_and_teacher_labels()
        student.train()
        optimizer.zero_grad()
        loss = mean_square(student(batch.RGB, batch.actor_obs) - batch.teacher11)
        loss.backward()                             # Full 12 x 81 population
        clip_global_gradient_norm(student, 1.0)
        optimizer.step()                            # Exactly one update
        save_and_register_immutable_model_optimizer_checkpoint()
        log_train_MSE_gradient_norm_checkpoint_and_timing()
        activate_new_checkpoint_at_the_completed_window_barrier()

    evaluate_declared_final_checkpoint_on_the_fixed_success_panel()
    next_round_starts_from_this_rounds_final_iteration_50_checkpoint()
```

One persistent Isaac/history/RNN/RNG lifecycle spans the windows within a
collection segment. An 81-step boundary is not permission to reset all lanes.
Real episode termination resets only the affected lanes. A DAgger window may
cross episode boundaries, unlike an emitted BC video chunk; per-row episode
IDs, step indices, `done` and `timeout` must preserve those boundaries.

```text
one actor vector step           = 6 physical environment transitions
one complete two-actor window   = 2 x 6 x 81 = 972 transitions
one learning iteration          = 12 complete chunks + 1 optimizer update
50 current-only iterations      = 48,600 transitions + 50 optimizer updates
```

The student stays frozen during a collection window, then changes before the
next window. It is neither updated per simulator step nor held unchanged
until the entire DAgger round ends. The current candidate uses LR 0.0002,
gradient clip 1.0, one full 12 x 81 learner batch, one minibatch and no activation
recomputation. All renderer and teacher/Homie weights stay frozen.

MSE is the pre-update prediction error on the newly collected training batch.
It need not match BC validation MSE because data, observation source and
train/eval mode differ. This does not prove a discrepancy is harmless: check
checkpoint identity, normalization, camera mapping and supervision alignment
before attributing a gap to distribution shift.

### Historical Beta replay and round continuity (superseded)

This subsection records the older fixed-50-iteration Beta protocol. For new
runs, the October 15 override and `pipeline/dagger/MSE_VALIDATION_PROTOCOL.md`
take precedence: select on independent held-out MSE, carry the selected
checkpoint **and optimizer state** into the next round, and measure success
only after selection. Success-test results must not choose the next parent.
The numerical B=6/B=12 example above is likewise historical; it is not a
current Leo memory qualification or a command to scale the native owners.

The measured beta candidate used **current-window data only**. The repository
also implements an opt-in replay extension for retaining corrected BC and
prior DAgger support. Its existing schedule assumes a total update batch of
28 chunks. It is not a certified B12 production configuration and is not
implicitly enabled by this document. Freeze an explicit update population and
mixture before promoting beta production; changing replay changes the
algorithm, not just storage.

The user-requested round budget is 50 updates per round, with success
evaluation at final iteration 50. Historical round 1 additionally evaluated
at 40. That extra validation is a declared schedule choice, not a requirement
to restart or shorten a round. BC uses dataset epochs; DAgger uses collection
iterations and rounds. Do not call a simulator step an optimizer step/epoch.

[Next-round materialization](../pipeline/dagger/materialize_next_round.py)
uses the preceding final learner candidate. Validation may separately record
the best-success checkpoint; it must not silently replace the requested
iteration-50 warm start. Each round gets a distinct W&B name reflecting its
teacher percentage.

## Cross host boundary and recovery rules

| Route | Minimal v3 payload |
|---|---|
| RTX to H200, all lanes | Run/round/window/episode/step identities, small metadata, `actor_obs[113]` float32 and `teacher_action[26]` float32 |
| Generated-control lane | Three `depth_ctrl` uint8 arrays at 240 x 416 and three `rgb_robot` uint8 arrays at 240 x 416 x 3 |
| Ray-traced lane | Three `raytraced_rgb` float32 arrays in `[0,1]` at 108 x 192 x 3; no generated controls |
| Anchor registration | Three uint8 anchor RGB tiles at 240 x 416 x 3, sent for required cache registration rather than every step |
| H200 to RTX | Selected 26-dimensional action and bound selection/identity receipts |
| RTX commit to H200 | Outcome and exact prepared-step identity, not another image payload |

The lean codec uses binary arrays and fast lossless compression; it avoids
base64 image expansion. It omits per-step raw metric/quantized depth and
privileged teacher/critic/Homie observation archives. Homie's 15 action values
are already available in `teacher_action[11:]`. H200 stores the exact policy
RGB used for inference along with matching state, teacher label, selected
behavior and outcome; RTX does not upload a second learner-complete NPZ.

Current ports are loopback-only and authenticated through SSH forwarding:

| Port | Owner |
|---|---|
| 18610 | Coordinator |
| 18611 | Trajectory service |
| 18612 | Checkpoint registry |
| 18613 | Learner job service |
| 18614 | Success-evaluation ledger |
| 18620 through 18623 | Four render/student gateways |
| 18500 through 18503 | Four Image3F workers |

Non-negotiable recovery rules:

An abandoned, next-round `PREPARE` may use the opt-in
[round-boundary admission](../pipeline/deployment/ROUND_BOUNDARY_RECOVERY.md)
only after completed learning/success/W&B receipts, unchanged idle resident
generations, an absent allocation, and zero next-round activity are verified.
It preserves the shared ledger/checkpoint chain and creates a fresh isolated
infrastructure attempt; it does not restore simulator state or repeat completed
training. Mid-window and uncertain native failures remain non-resumable.

- Only the RTX native owner executes `env.step` and only after every local
  lane has a valid selection for the same step/checkpoint.
- Prepare and commit retries preserve route-scoped idempotency keys and
  payload identities. Identical retry replays the recorded response; changed
  payload under the same key is an error.
- A lost commit response must not repeat the physical action.
- H200 assembles learner rows only from committed transitions. It must not
  learn from an incomplete/abandoned window.
- An uncertain simulator process restart cannot be recovered by reloading a
  student checkpoint alone: physics, RNG, cameras and controller histories
  are state too. Preserve evidence and use a declared fresh continuation.
- Keep each concurrent framed connection isolated/serialized correctly; do
  not share an unguarded Image3F stream between inference and health probes.
- Anchor/checkpoint caches must bind exact content identities. Every next
  window must use one agreed student checkpoint across workers and actors.
- Essential identity, finite/shape and exactly-once checks remain. Do not put
  full dataset hashing, repeated raw-depth restacking, or heavy duplicate RTX
  chunk compression in the per-step critical path.

Code owners: [wire contract](../pipeline/dagger/services/batch_contract.py),
[render/student inference](../pipeline/dagger/services/render_student.py),
[RTX owner/orchestration package](../pipeline/dagger/rtx/code/src/dagger_rtx/),
[learner](../pipeline/dagger/services/raven_learner_worker.py), and
[production controller](../pipeline/dagger/production_controller.py).

## Success evaluation versus offline validation

Offline BC validation is held-out action MSE on saved rendered chunks, on
H200 without Isaac. Closed-loop success evaluation runs a frozen policy
through complete native Isaac episodes, with no optimizer or backpropagation.

```python
student.eval()
disable_gradients()

for case in immutable_256_case_panel:
    reset_to_the_case_using_the_qualified_native_protocol()
    goal_reached = False
    while not episode_done:
        rgb = observation_for_explicit_evaluation_mode()
        student11 = student(normalize_once(rgb), actor_obs)
        action26 = concat(student11, frozen_homie_action)
        outcome = isaac.step(action26)              # 100 percent student behavior
        goal_reached |= outcome.infos["grasp_goal_reached"]
        episode_done = outcome.done
    record_this_cases_first_completed_episode_once(success=goal_reached)

success_rate = successful_completed_cases / completed_cases
require_exactly_256_registered_cases_before_final_reporting()
```

The metric is the OR-latched native `grasp_goal_reached` Boolean, not a visual
judgment or MSE threshold. A goal reached earlier remains latched if the
episode later terminates differently. The geometry/task conditions producing
that flag belong to the pinned native environment; do not invent a substitute
distance threshold.

The corrected evaluation adapter records the native goal **before automatic
reset** and publishes that transition snapshot as `infos["grasp_goal_reached"]`.
Its accounting version is `roborender.raven.pre_reset_goal.v1`. The new episode's
buffers remain cleared, preventing success from leaking across resets. This is
an explicit correction to Raven's inherited terminal-frame export ordering,
not a change to the success criterion. Do not silently compare or combine
uncorrected historical scores with corrected scores.

Evaluation modes are explicit and reported separately:

- **Ray-traced:** RTX owns Isaac, Homie and local student inference after
  staging the candidate; Image3F is absent.
- **Generated:** RTX owns Isaac and native policy evaluation; H200 Image3F
  supplies generated observations. This is the photorealistic-observation
  evaluation path, not a teacher/student behavior mixture.
- **Privileged teacher oracle:** teacher plus Homie control Isaac, providing
  an environment/teacher reference independent of the student's RGB branch.

The warm paired evaluator uses one native B5 instance for reference and
candidate panels. Each checkpoint runs the complete declared panel, with the
same reset protocol and byte-exact paired physical/state observations.
Checkpoint switching occurs only between panels, with zero optimizer steps.
The evaluator is released before both training actors restart: the current
Leo host-RAM admission does not include a concurrently resident evaluator.
See the warm-evaluation section of the end-to-end smoke runbook for receipts
and deployment gates.

Compare BC and DAgger on the same qualified scenario/reset manifest, camera
calibration and observation mode. Do not compare a historical random
300-episode result with a fixed 256-case result or one mode against the other.
The final test remains separate from repeated checkpoint selection.

The [three-view fixed-panel protocol](../pipeline/evaluation/THREE_VIEW_FIXED_PANEL.md)
requires certified camera geometry and native reset/terminal behavior. The
current calibration and beta runtime are not yet certified. Raven's reported
success percentages and earlier low local scores are not certified beta
baselines. Settling the evaluation protocol remains an explicit task.

## Logging and operational handoff

Log collection phase start/end timestamps separately from learner telemetry:
Isaac capture/teacher, request encode/transfer, H200 queue/render/student,
behavior selection, physical step, commit, seal/publish, learner, checkpoint
activation and evaluation. Do not sum nested spans as exclusive percentages.

W&B project is `dagger`. BC uses run name `BC`; DAgger names identify round,
teacher percentage, deployment and unique run identity. Record config/source
hashes, seeds, checkpoint parents, model/engine identities and observation
mixture. Record `train/iteration_mse` once per real DAgger optimizer update,
BC `train/mse_step` per update, BC `validation/mse` per epoch, and success
numerator/denominator plus `validation/goal_reached_rate` per completed panel.
Collection-only runs have no training MSE. Verify online W&B authentication
before claiming a live dashboard exists; offline files are not online logs.

The shared repository is the source of truth after reviewed commit/push/pull.
Until then verify matching source manifests rather than claiming identical
Git commits. Never copy credentials, `.git`, runtime ledgers or GPU-specific
engines as ordinary source synchronization. Do not reset or overwrite dirty
worktrees to force agreement.

The user-approved migration policy is **required working RTX assets on beta,
historical/reusable archives on H200 shared storage**. Complete archival
inventory and beta-owned authenticated access before declaring alpha
unnecessary. Beta authentication is to be supplied by the user; do not copy
alpha's personal API credentials or delete alpha originals on the strength
of this document.

## Updating this contract

When changing the pipeline, update this revision/date, the relevant
machine-readable profile and component docs together. Record whether a
change affects scientific behavior or only transport/performance. Keep
completed evidence separate from planned tests; a CPU test, one-step test,
full-window test and long-duration production test prove different things.

Preserve these invariants across performance and migration work:

1. Three synchronized policy views with one normalization and correct
   stage-specific renderer mapping.
2. Teacher labels and actor states from the same pre-action state as the RGB.
3. Exactly 11 learned action outputs and an unchanged 15-dimensional Homie
   tail; no accidental supervision on another slice.
4. Native ray-traced augmentation and random initial episode lengths in
   training; explicit separately frozen evaluation semantics.
5. Full configured training-batch statistics, not silent smaller BatchNorm
   populations or a legacy B28 replay schedule substituted for B12.
6. One agreed rollout checkpoint per window, one physical step per decision,
   and activation only at the completed learner barrier.
7. No loss of checkpoint/optimizer/data provenance or false claim of
   success-rate improvement from MSE alone.

Raven's reported teacher-only BC experiment and the requested mixed-behavior
DAgger experiment are different experiments. Three policy views, distributed
execution, the beta simulator port, optional replay and fixed-panel evaluation
are explicit extensions or qualification boundaries. Matching component code
does not establish identical state distributions or headline success rates.
