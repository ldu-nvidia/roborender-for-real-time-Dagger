# Vision-student training without ray tracing

Two ways to train the RGB student policy when ray-traced rendering is too slow to
sit inside the training loop:

- **A. Offline BC** — pre-render a fixed dataset of generated frames, then behaviour-clone on it.
- **B. Online image-model DAgger** — a warm Image3F server relabels every observation
  in the rollout, so the student trains on generated pixels while the teacher supervises actions.

Both produce a student for `walk_stand_place_grasp_turn_homie` (bottle → tray).
Code lives in `GR00T-VisualSim2Real/`; launchers and assets live here in `fast_roborender/`.

## Setup

Three separate Python environments, because the trainer and the renderers cannot share one
(the render bundles pin their own torch/TensorRT builds):

| Environment | Built by | Used for |
|---|---|---|
| conda `viral` | `scripts/setup/setup_viral_env.sh` | Isaac Sim, IsaacLab, `gr00t` — all training and eval |
| `roborender-acceleration/.venv` | the bundle's `bin/create_baremetal_env.sh` | teacher video model, LADD students, offline rendering |
| `<image3f bundle>/.venv` | `scripts/server/image3f_ft_setup.sbatch` | Image3F per-frame renderer (DAgger) |

```bash
# 1. trainer environment (~30 min; clones IsaacLab, installs Isaac Sim 5.1)
bash scripts/setup/setup_viral_env.sh

# 2. Image3F bundle: venv, ONNX export, TensorRT engine, doctor + smoke test.
#    Must run on the GPU you will deploy on — engines are arch- and TRT-version-specific.
sbatch scripts/server/image3f_ft_setup.sbatch

# 3. verify everything before submitting real jobs
bash scripts/setup/check_env.sh
```

`check_env.sh` is read-only and prints PASS/WARN/FAIL for the trainer env and its imports,
all three render environments, the Image3F adapter and foundation weights, TensorRT engines
present per GPU arch, the anchor pool and dataset, the baseline checkpoint and config that
`eval_policy.sbatch` stages against, any running render servers (flagging **stale endpoint
files**, whose host no longer has a server), and the viscam Slurm account. WARN means "fine
unless you need that stage". Exit status is non-zero only on hard failures.

### Setup gotchas

- **Engines are per GPU arch and per TensorRT version.** `sm120` (RTX PRO 6000), `sm89`
  (L40S), `sm86` (A40) each need their own build, and a LoRA change requires a fresh ONNX
  export. Nothing is shipped prebuilt — `assets/generated/` starts empty by design.
- **The bundles hash-verify their own source tree.** Never leave edits in a bundle: patch,
  run, restore. `PYTHONDONTWRITEBYTECODE=1` is required, since stray `.pyc` files fail the
  release-tree check, and the package-stage doctor demands an empty `assets/generated/`
  (clear any orphaned staging after a failed export).
- **Extract archives inside a Slurm job.** The login-node process reaper kills long
  unzips (a 22 GB extract died with exit 137).
- The Image3F bundle ships only the LoRA delta; foundation weights are attached via
  `ROBORENDER_MODEL_BASE`, pointing at the older portable bundle's `assets/models`.
- Model load is ~110 s with a warm page cache but up to ~40 min cold from NFS, which is why
  the render server is a persistent job rather than something each run starts.

## Repository layout

```
README.md                     this file
run_raven_server.sh           render-server entrypoint (bundle-relative; stays at root)

scripts/
  setup/    setup_viral_env.sh  check_env.sh
  scaling/  scale_plan.py  image3f_probe.py  install_gr00t_integration.py
            submit_production.sh
  data/     collect_data.sbatch  offline_render.sbatch  render_shard.sbatch
            offline_render_dataset.py  renormalize_depth.py
  anchors/  anchor_pool.sbatch  render_anchor_pool.py  rich_prompts.py
            teacher_anchor_render.sbatch  render_teacher_fullres.py  videogen_prompts.py
  train/    offline_bc_train.sbatch  dagger_train.sbatch  image_dagger_*.sbatch
            dagger_scale_256x300.sbatch
  eval/     eval_policy.sbatch  eval_latest.sbatch  eval_student.sbatch
  server/   image3f_server.py  image3f_server.sbatch  raven_server.sbatch
            image3f_ft_setup.sbatch  image3f_ft_eval.sbatch

tools/      benchmarks, sample/preview builders, the 5-vs-50-step A/B (step_ab*.py)
integrations/gr00t/
            versioned Image3F client and strict private-trainer source transform
tests/      login-node unit tests for scale math and lag-reference semantics
archive/    superseded one-off diagnostics and earlier experiments

GR00T-VisualSim2Real/         trainer, configs, simulator (own git repo, branch
                              videogen-image-dagger)
roborender-acceleration/      LADD student bundle + TensorRT engines (own repo)
```

Everything else at the top level is gitignored run-time state or data:
`videogen_dataset/`, `teacher_fullres/`, `logs_archive/`, `image3f_queue*/`,
`*_endpoint*.txt`, `logs_rl/`.

## Results

All evaluated on **ray-traced** observations under the **offline-BC eval config** (see
[Evaluation](#evaluation-read-this-first) — this qualifier is load-bearing).

| Policy | Episodes | Success | Reward |
|---|---|---|---|
| Offline BC (`videogen_offline_bc`, step 11000) | 150 | 57.3% | −993 |
| Round-3 image DAgger (walled env — see below) | 150 | 58.7% | −547 |
| **Round-4 image DAgger** (`image_dagger_r4`, iter 13227) | **300** | **69.3%** | **−426** |

Round 4 is +12.0 over the BC baseline (SE 2.7, p ≈ 0.013). Final `dagger_bc_loss` 0.134.

Generated observations are **policy-equivalent to ray tracing**: a policy that never
trained on generated frames scores 59.3% on them vs 57.3% ray-traced (150 eps each).
The renderer is not a bottleneck for this task.

## Which renderer to use

| Model | Path | Granularity | Speed | Use for |
|---|---|---|---|---|
| **50-step teacher video model** | `renderscale/Sim2RealGen_Video_Model/lora_ckpts/Wan2.1-Fun-V1.1-1.3B-Mixed-DROID-AGIBOT-3View-DualControl-PrevFrame-droidnew_continue_lora/step-5294.safetensors` | 81-frame 3-view clip | ~32 s/clip (~400 ms/frame) | **A** dataset rendering; **B** anchor pool |
| LADD 5-step student, 3-view | `roborender-acceleration/assets/student/raven_native3view_ladd_5step_iter5250.pth` | 81-frame 3-view clip | 6.9 s/clip (~85 ms/frame) | **A** when throughput matters more than texture |
| LADD 5-step student, 2-view | `roborender-acceleration/assets/student/raven_ladd_5step_global_best_iter5250_bf16.safetensors` | 81-frame 2-view clip | TRT-accelerated (3.37× smoke) | 2-view tasks only — no wrist-pair support |
| **Image3F finetune (image model)** | `/vision/u/ravenh/roborender_image_model_accelerated`, adapter `rgb030-depth030-step-5000.safetensors` | **single 3-view frame** | ~192 ms/frame warm | **B** — the only option for in-loop relabelling |

**A (offline BC) → 50-step teacher.** You are rendering whole clips ahead of time, so
per-frame latency does not matter and clip-based video models give the best quality. The
production dataset was rendered this way. Swap in the 3-view LADD student if you need ~4.6×
throughput and can accept washed-out object textures (PSNR 17.6–20.5 vs teacher).

**B (online DAgger) → Image3F finetune.** DAgger has to relabel *one step at a time*, which
rules out every clip-based model: they cannot render a single frame, and 32 s/clip inside the
rollout loop would be ~200× too slow. Image3F is the only per-frame renderer, and at ~192 ms
per 3-view frame it sustains ~145 s/iteration at 16 envs across 3 servers.

Note that **B still needs the teacher model**, indirectly: Image3F runs in FT-FIXED mode,
where every frame is conditioned on a fixed teacher-rendered RGB triplet for that episode.
That is what `scripts/anchors/anchor_pool.sbatch` produces (2048 anchors, `scripts/anchors/rich_prompts.py`). Conditioning
Image3F on its own previous outputs instead collapses around frame 20 — do not chain.

Image3F specifics: 5 denoising steps, CFG 1.0, TeaCache 0.3, TensorRT on the 30-block DiT
stack, batch size fixed at 1 (the DiT is compute-bound there, so scale by adding servers, not
by batching). More denoising steps make it *worse* for this use — 50 steps drifts off the
anchor's colours and materials and costs ~2.8 dB of teacher fidelity for 8× the latency.

## Data generation (shared by A, and by B's anchors)

**1. Collect depth/mask chunks.** Roll out the teacher and save the control signals the
renderer needs — per-view depth, robot-body masks, teacher actions — in 81-frame chunks
(the video model's clip length).

```bash
sbatch scripts/data/collect_data.sbatch                  # +exp=loco_manip/wsdpt_videogen_collect
                                            # NUM_ENVS (default 32)
```

Chunks land in `videogen_dataset/` as `chunk_<jobid>_<NNNNNNN>.npz`, each holding
`depth_raw__<view>`, `depth_ctrl__<view>`, `rgb_robot__<view>` for `head_color`,
`hand_left`, `hand_right`, plus the prompt. The production set is ~2000 chunks / 31 GB.

**2. Render RGB.** A standing render server converts depth+mask into photoreal RGB.
Start the server first, then stream the dataset through it:

```bash
sbatch scripts/server/raven_server.sbatch                  # or scripts/anchors/teacher_anchor_render.sbatch for the 50-step teacher
sbatch scripts/data/offline_render.sbatch                # -> scripts/data/offline_render_dataset.py --follow
                                            # RENDER_VIEWS, RENDER_ARGS
```

`--follow` keeps the renderer consuming chunks as collection produces them, so steps 1 and 2
can overlap. Scene text comes from `render_prompt.txt` (single fixed scene) or
`scripts/anchors/videogen_prompts.py` / `scripts/anchors/rich_prompts.py` for randomized scenes. `scripts/data/render_shard.sbatch` splits
the dataset across several servers.

Depth is normalized with the `h5_v4` recipe before it reaches the model
(`scripts/data/renormalize_depth.py`); per-view clip ranges differ between the head and wrist cameras.

## A. Offline BC

Behaviour-clone on the rendered dataset — no renderer in the loop, so this is the cheap path.

```bash
sbatch scripts/train/offline_bc_train.sbatch              # +exp=loco_manip/wsdpt_videogen_offline_bc
                                            # NUM_ENVS (default 16)
```

Checkpoints land in `logs_rl/videogen_offline_bc/<run>/`. This produced the 57.3% baseline.

## Anchors (FT-FIXED conditioning)

Image3F never renders from depth alone. Every request carries a `previous_rgb` triplet that
fixes the scene's appearance — colours, materials, lighting — while depth and the robot mask
supply geometry. In **FT-FIXED** mode the *same* teacher-rendered triplet is passed on every
frame of an episode. This is the model's training contract (the finetune was trained with a
fixed-lag-256 reference policy, where `t < 256` always references frame 0), and it is what
keeps appearance stable across a rollout.

The alternatives both lose: chaining on the model's own previous output collapses around
frame 20, and dropping the reference entirely is out of contract for this checkpoint (its
view-dropout only ever dropped *one* view, never all three).

**The pool.** `/vision/u/ravenh/anchor_pool_2048` — 2048 anchors, each an independent scene
rendered by the 50-step teacher:

```
anchor_00000.npz
  gen__head_color   (2, 240, 416, 3) uint8   # frames [0, 40] of the teacher clip
  gen__hand_left    (2, 240, 416, 3) uint8
  gen__hand_right   (2, 240, 416, 3) uint8
  frames            (2,) int64               # [0, 40]
  prompt            <U541                    # the scene description used to render it
  src               <U26                     # source depth chunk
```

**Building it.** `scripts/anchors/render_anchor_pool.py` is shardable; `scripts/anchors/anchor_pool.sbatch` starts a teacher
server (LoRA path is required — the server dies instantly without it) and renders a range:

```bash
# START/END/OUT are read from the environment; the render server's port is derived
# from the job id inside the script, so parallel shards never collide.
for SH in 0 1 2; do
  sbatch --export=ALL,START=$((SH*683)),END=$(((SH+1)*683)),OUT=/vision/u/ravenh/anchor_pool_2048          scripts/anchors/anchor_pool.sbatch
done
```

Prompts come from `scripts/anchors/rich_prompts.py::build_rich_prompt(seed)`, which composes walls (16) ×
floors (14) × tables (14) × objects (16) × containers (12) × lighting (6) ≈ 29M combinations,
each with explicit texture wording ("industrial exposed red-brick walls with worn white paint
patches", "translucent amber plastic bottle with a white flip-top cap"). At 2048 draws
repeats are effectively nil, so anchors are the pipeline's main source of visual diversity —
this is where domain randomization enters the generated-observation path.

**Using it.** `videogen.anchor_pool_dir` points at the pool; `videogen.anchor_index` selects
which stored frame becomes the reference. One anchor is drawn per environment and resampled
on `done`, so each episode gets its own scene (`_id_sample_anchor` in
`distill_trainer_videogen.py`, which accepts both this pool's `frames`-array layout and the
older `*_teacher3v.npz` direct-index layout).

**Which frame — 0 or 40?** Round 4 used `anchor_index: 0` and that is the safe default,
matching the training contract exactly. But frame 0 is content-poor for the wrist cameras at
episode start (the hands have not reached the table yet, so those references are nearly
empty), and a content-rich anchor measurably improves colour fidelity in precisely those
views. Frame 40 is stored alongside frame 0 for exactly this reason. It is slightly
off-contract but stable in practice, and it is the cheapest untried lever if you want more
from the renderer — the pool already contains the data, so switching costs one config line.

## B. Online image-model DAgger

The student acts, the teacher supplies the supervision, and Image3F relabels observations
inside the rollout — so the student sees generated pixels on the states its own policy visits,
which offline BC cannot give you.

```bash
# 1. anchor pool (see Anchors above) — only needed once
sbatch scripts/anchors/anchor_pool.sbatch

# 2. warm renderers: ~1 per 5 relabelled envs, each with its own endpoint file and queue
for i in '' _2 _3; do
  sbatch --time=3-00:00:00 \
    --export=ALL,ENDPOINT_FILE=$PWD/image3f_server_endpoint$i.txt,IMAGE3F_QUEUE=$PWD/image3f_queue$i \
    scripts/server/image3f_server.sbatch
done

# 3. train
sbatch --export=ALL,MIX=0.5 scripts/train/dagger_train.sbatch
```

### Observation mixing

`MIX` is the fraction of environments left on **ray-traced** observations
(`videogen.raytraced_mix_prob`):

| `MIX` | Regime | Renderers for 16 envs | Notes |
|---|---|---|---|
| `0.0` | **fully image-generated** | ~3 | every env relabelled; trains purely on the generated distribution |
| `0.5` | **mixed** (default) | ~2 | what round 4 used → 69.3%; hedges against renderer artefacts |
| `1.0` | fully ray-traced | 0 | no relabelling; plain DAgger, useful as a control |

`scripts/train/dagger_train.sbatch` validates `MIX ∈ [0,1]`, discovers endpoint files whose host still runs
an `i3f_warm_server` (a stale endpoint otherwise kills the run hours in), refuses to start if
relabelling is requested with no live renderer, and warns when there are fewer servers than
`ceil(envs × (1 − MIX) / 5)`. Too few servers does not fail — it serialises the rollout, so
watch `Iteration time` (expect ~145 s at 16 envs / 3 servers / `MIX=0.5`).

Other variables: `CKPT` (warm start, defaults to the round-4 final), `NUM_ENVS` (16),
`ANCHORS`, `ANCHOR_IDX`, `EXP`/`PROJ`. Run names default to the mix
(`dagger_fullgen`, `dagger_mix0.5`, `dagger_fullrt`).

### Config that matters

`gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml`:

| Key | Value | Why |
|---|---|---|
| `algo.config.value_loss_coef` / `entropy_coef` | `0.0` | pure BC; RL terms with an untrained value head corrupt the policy |
| `videogen.mode` | `image_dagger` | per-step 3-view relabel in `distill_trainer_videogen.py` |
| `videogen.raytraced_mix_prob` | via `MIX` | see above |
| `videogen.image3f_endpoint_files` | one per renderer | envs are sharded round-robin across them |
| `videogen.anchor_pool_dir` / `anchor_index` | `anchor_pool_2048` / `0` | fixed anchor per episode; chaining on own outputs collapses |
| `num_steps_per_env` | `81` | one BC update per video-model clip length |

Warm-start from the previous round's `last.pt`: round 4 started from round 3 and reached
`dagger_bc_loss` 0.146 within ~900 iterations (round 3 ended at 0.67, and finished at 0.134).

### Production scale: 256 environments × 300 rollout steps

The production launcher uses **eight synchronized trainer ranks with 32 Isaac environments
per rank**. `num_envs` is rank-local in GR00T, so passing 256 to every rank would accidentally
create 2,048 environments. Each outer rollout-and-learning iteration collects 76,800
transitions; rank sharding also avoids the unsafe single-L40S rollout allocation.
Accelerate/DDP synchronizes the
student's supervised DAgger gradients after each rank collects its local trajectories.
With the pinned production settings (`num_learning_epochs=1`, four minibatches), that outer
iteration performs four optimizer steps, each covering 19,200 global transitions.

The production-safe default is `MIX=0.0`: all 256 environments render every simulator tick,
requiring 52 independent RTX PRO 6000 Image3F servers at five environments per server. This
guarantees contiguous lag histories. The previously validated `MIX=0.5` recipe needs only 26
servers, but it is gated behind `ALLOW_EXPERIMENTAL_MIXED_LAG=1` until Raven's private trainer
is verified to use an episode-sticky generated cohort; per-tick Bernoulli mixing would make
`t-256` histories discontinuous. Endpoints live in a run-scoped selected manifest and are
actively pinged. Training refuses to start until every required server advertises protocol 2
and a maximum reference lag at least as large as the requested lag.

The 300-step horizon crosses Image3F's training-time reference boundary. The production client
and server therefore implement the exact policy: use the episode anchor at `t < 256`, then use
the generated RGB from `t-256`. History is bounded and held server-side; environment streams
remain sticky to endpoints. Legacy 81-step launchers keep their stateless behavior unless
`IMAGE3F_REFERENCE_LAG` is set.

The core trainer is a separate, gitignored Raven GR00T fork. Public NVlabs GR00T does not have
`distill_trainer_videogen.py`, `image3f_frame_client.py`, or `wsdpt_image_dagger.yaml`. The
installer patches the known parallel trainer to pass explicit episode/step IDs and installs
the versioned client. Run it only after checking out that exact fork, review both changes, then
commit them in the GR00T repository so production sees a clean, pinned revision:

```bash
python scripts/scaling/install_gr00t_integration.py \
  --gr00t-root GR00T-VisualSim2Real --install

# inspect the exact 256x300 capacity calculation
python scripts/scaling/scale_plan.py

# validate commands without submitting jobs
DRY_RUN=1 bash scripts/scaling/submit_production.sh

# submit 52 renderer array elements plus the 8-rank trainer (MIX=0 default)
bash scripts/scaling/submit_production.sh

# pinned production campaign: 256 environments, 300-step rollouts, 8-hour wall time
bash scripts/scaling/submit_8h_256env.sh
```

The eight-hour value is the trainer allocation wall time, not a promise of eight full hours of
optimizer execution: endpoint readiness and final cleanup happen inside that allocation. The
trainer is submitted with an `after:<renderer-array>` dependency so Slurm does not allocate its
eight trainer GPUs until every renderer-array task has at least started; it still verifies every
endpoint before the first rollout. Renderer workers use a longer lease as scheduling/startup
headroom, but that lease is a bounded safeguard rather than a gang-scheduling guarantee. The
launcher records renderer and trainer job IDs incrementally in `slurm_jobs.json`; if trainer
submission fails it cancels only the renderer array it just created. Once trainer submission
succeeds, that renderer ID is passed to the trainer, which cancels the fleet on every success or
failure exit. Slurm sends the batch shell `USR1` five minutes before the trainer limit; the shell
stops the distributed step, retains the latest five-iteration periodic checkpoint, drains
telemetry, and records a nonzero outcome instead of waiting for a hard time-limit kill. A separate
`afterany:<trainer_job>` cleanup watchdog also owns the renderer job ID; this covers a trainer
cancelled while still pending, which never enters its batch shell and therefore cannot run that
EXIT trap.

Useful overrides are `GLOBAL_NUM_ENVS`, `ROLLOUT_STEPS`, `TRAINER_NODES`, `MIX`,
`REFERENCE_LAG`, `CKPT`, `NUM_ROLLOUT_ITERATIONS`, `PROJ`, and `EXP`. The trainer launcher logs a
machine-readable scale plan, both repository revisions, the checkpoint, endpoint fleet, and
rank-specific stdout/stderr under its run directory. It uses microbatch 1 with gradient
accumulation 8 by default, preserving the effective local mini-batch while limiting 300-frame
sequence memory. `ROLLOUT_STEPS=300` is the collection horizon of each outer rollout
iteration; it is not an optimizer-batch size and is distinct from run length.
`NUM_ROLLOUT_ITERATIONS` defaults to 1 for a bounded scale-validation run and maps to the
trainer's legacy `algo.trl.num_total_batches` setting. `NUM_TOTAL_BATCHES` remains a deprecated
environment alias; setting both names to different values is a fatal error.

Production behavior is explicit and recorded in both manifests. The default is
`BEHAVIOR_SOURCE=student`, which pins `rollout_with_teacher_num_steps=0`,
`enforce_teacher_rollout=false`, and `TEACHER_ROLLOUT_RATIO=0.0`. An intentional mixed run must
set `BEHAVIOR_SOURCE=mixed` and a ratio strictly between zero and one; a teacher-only run must
set `BEHAVIOR_SOURCE=teacher` and ratio `1.0`. Inconsistent combinations fail before any Slurm
submission. `NUM_LEARNING_EPOCHS=1` and `VG_PRIMARY_VIEW=head_color` are also pinned explicitly.
The supported primary views are `hand_left`, `hand_right`, and `head_color`. Image3F still
generates all three synchronized views, but this trainer supplies only the selected primary
view to the actor. The GR00T integration patch installs matching runtime assertions inside the
trainer, so a composed Hydra configuration that disagrees with the launch manifest fails before
its first rollout. The production command also pins `value_loss_coef=0.0` and
`entropy_coef=0.0`; both source-config inspection and the composed trainer assert exact numeric
zero, so a prefix such as `0.1` cannot pass as pure behavior cloning.

### Explicit production gates

| Gate | Must pass before/while running |
|---|---|
| Source and assets | Clean, run-pinned `fast_roborender` and private GR00T revisions; versioned Image3F client/telemetry integration; launch-time checkpoint and reference-config SHA-256 values unchanged |
| Scale and behavior | 8 ranks × 32 environments = 256; 300-step rollout; 52 renderers; `MIX=0`; lag 256; student behavior; one learning epoch; pure DAgger BC |
| Renderer provenance | All 52 live endpoints report the exact fine-tuned release/adapter SHA, five-step sampler, TeaCache 0.3, CFG 1.0, CUDA graph setting, and a load-verified TensorRT engine compatible with the serving GPU/software stack |
| W&B observability | Online personal workspace authenticated as `ldu@nvidia.com`; stable resumable sidecar run; scalar, condition/generated image, and MP4 producers ready; repeated upload/encoding errors terminate training |
| Runtime semantics | The composed private trainer agrees with every behavior, primary-view, epoch, and loss-coefficient assertion before its first rollout |
| Scientific result | Baseline/early/mid/final fixed-seed evaluation is complete; the final policy improves success by at least 2 absolute points and the paired 95% bootstrap interval has a lower bound above zero |

The full `MIX=0` topology asks Slurm for 8 L40S trainer GPUs plus 52 RTX PRO 6000
renderer GPUs. The orchestration repository, two-GPU workstation, or an otherwise idle cluster is
not by itself sufficient: the Raven private GR00T trainer/config, actor checkpoint, Image3F model
assets, compatible per-GPU TensorRT builds, and W&B authentication must also be present. A failed
gate never becomes a nominal production run; transactional submission cancels only jobs created by
that submission.

### W&B production telemetry

Production submission fails closed unless W&B is online, the authenticated account email is
`ldu@nvidia.com`, and the resolved entity is that account's personal entity rather than a team.
The trainer's rank-zero process logs optimization and environment scalars, including DAgger BC
loss, learning rate, rewards, episode lengths, FPS, collection time, and learning time. A separate
rank-zero sidecar uploads stable telemetry files from `telemetry/scalars`, `telemetry/images`, and
`telemetry/videos` on a shared `trainer_step` axis; it is restart-safe and does not log partially
written media. Production checkpoints are saved every five rollout iterations so early and late
policy behavior can be evaluated with fixed seeds and the same expert/reference configuration.
The media watcher is a production gate: if it exits while the trainer is live, the distributed
step is stopped. `RUN_DONE` is written only after a successful trainer and telemetry drain;
failures write `RUN_FAILED` with the preserved trainer exit status.

### Fixed-seed policy-improvement gate

The pinned eight-hour launcher also schedules a ray-traced policy evaluation with
`afterany:<trainer_job>`, so a useful checkpoint can still be measured when the eight-hour
wall-time ends the trainer. It selects four distinct policies—warm-start baseline, first saved
checkpoint, midpoint checkpoint, and latest checkpoint—and evaluates each on the same eight
Isaac seed blocks with 32 environments per block. Each environment contributes exactly one
episode, giving 256 matched episodes per checkpoint without letting fast-terminating policies
advance the reset RNG to different scenarios. All checkpoints use the same offline-BC reference
config. The baseline checkpoint and reference-config hashes are captured before training;
both the trainer and delayed evaluator fail if those paths later resolve to different bytes.
Evaluation atomically rebuilds each staged checkpoint, removes only environment, optimizer,
and scheduler state, and records the actual staged checkpoint/config hashes before reset. A strict
in-tree adapter captures the mapping returned by `trainer.eval()` and writes it atomically; the
gate does not rely on a particular private trainer variant happening to emit `metrics_eval.json`.
Generic bounded smoke runs automatically disable this statistical gate when their configured
run length cannot produce three distinct saved snapshots; the pinned campaign explicitly
enables it.

The machine-readable report is
`<training_run>/fixed_seed_eval/fixed_seed_improvement_report.json`. The gate passes only when
final-minus-baseline success is at least two absolute percentage points and the lower bound of
its paired seed-block bootstrap 95% interval is above zero. Reward and its paired interval are
reported as a diagnostic but are not silently mixed into the success criterion. Training loss
is explicitly excluded as evidence of closed-loop improvement. Baseline/early/mid/final success,
Wilson intervals, reward intervals, and the gate result are logged to a separate evaluation run
in the same personal W&B project. `slurm_jobs.json` starts with the evaluation-orchestrator ID;
the orchestrator atomically adds the four-task evaluation-array and report job IDs after Slurm
accepts them. Failure to submit that dependent gate rolls back both the newly submitted trainer
and renderer fleet, so an ungated production run is not left alive. The launcher transfers
ownership only after Slurm accepts the trainer, renderer-cleanup watchdog, and evaluation
orchestrator.

Before the full job, validate progressively: 4×4, 32×16, 64×75, 128×75, 256×75,
256×260 (the lag transition), then 256×300. Acceptance requires exact global transition
counts, finite BC loss, balanced endpoint request counts, zero render errors, correct anchor
reset/t−256 telemetry, and synchronized policy checkpoints across ranks.

## Evaluation — read this first

**`eval_agent_trl.py` loads `config.yaml` from the checkpoint's own directory.** A DAgger
run's config describes a different environment than the baseline's, so scoring a DAgger
checkpoint out of its run directory compares two different worlds and the number means
nothing. This invalidated three rounds of results before it was caught.

`scripts/eval/eval_policy.sbatch` exists so that cannot happen: it always stages the checkpoint beside an
explicit reference config.

```bash
# headline number: ray-traced obs, baseline config, 300 episodes
sbatch --export=ALL,CKPT=<run>/last.pt scripts/eval/eval_policy.sbatch

# quick check
sbatch --export=ALL,CKPT=<run>/last.pt,EPISODES=50 scripts/eval/eval_policy.sbatch

# generated obs (needs a live renderer)
sbatch --export=ALL,CKPT=<run>/last.pt,OBS=generated scripts/eval/eval_policy.sbatch
```

| Variable | Default | Notes |
|---|---|---|
| `CKPT` | *(required)* | checkpoint `.pt`; fails fast if missing |
| `EPISODES` | `300` | SE ≈ 2.7 pts. 150 gives ≈ 4.0 — too coarse for a 10-pt claim |
| `OBS` | `raytraced` | `generated` sets `IMAGE_DAGGER_EVAL_RELABEL=1` and forces `raytraced_mix_prob: 0.0` |
| `CONFIG` | per `OBS` | override the reference config |
| `ENDPOINT` | `image3f_server_endpoint.txt` | `OBS=generated` only |
| `NUM_ENVS` | `4` | |

The script also fails loudly on the traps that previously ate whole runs: a missing
checkpoint or config, and — for `OBS=generated` — an endpoint file whose host has no
`i3f_warm_server` running on it, which is what a stale endpoint looks like. It excludes
viscam10/viscam12 (NFS hangs), logs to `logs_archive/eval_policy_<jobid>.log`, prints a
one-line `RESULT ...` summary, and cleans up its staging directory.

Reference configs it stages against:

- `raytraced` → `logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml`
  — the environment every number in [Results](#results) was measured in.
- `generated` → `logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/config.yaml`
  — same environment plus the `videogen`/`image_dagger` block that relabelling needs.

**One eval saturates one renderer** (~52 episodes/h). Two evals sharing a server each run at
half speed and overrun the job limit with no output, since the success rate is only printed
at the end. Give each concurrent `OBS=generated` eval its own server and `ENDPOINT`.

## Gotchas

- **Scene walls must not be collidable.** `cameras.add_scene_walls` gives the head camera a
  real background instead of void, but collidable walls sit in the robot's walking path and
  cost ~40 points (16.7% collidable / 56.0% none / 60.0% visible-but-non-collidable). The
  `cameras.wall_collisions` flag (default `False`, `isaacsim.py`) keeps the backdrop and
  removes the obstacle. Leave it off.
- **Stale endpoints.** `image3f_queue*/READY` and `image3f_server_endpoint*.txt` persist
  across jobs. A readiness check that only tests file existence will fire on a dead server
  from a previous run. Verify the endpoint's hostname matches the server job's node.
- **NFS hangs.** viscam10 and viscam12 have wedged `cp` in `handle_async_copy` for an hour
  with the destination already fully written. `--exclude=viscam10,viscam12` for eval jobs;
  check `ps -o wchan` on the node when a job runs with an empty log.
- **Slurm.** `--nodelist=A,B` requests *all* listed nodes — use `--gres=gpu:<type>:1 --nodes=1`.
  `viscam-hi` preempts `viscam`/`sc-loprio`. Hydra needs a `+` prefix for keys absent from
  the base config. Training runs end in `TIMEOUT` at their limit; that is normal, and
  `--dependency=afterany` (not `afterok`) is what chains an eval behind them.
