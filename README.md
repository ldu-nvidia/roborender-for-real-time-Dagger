# Vision-student training without ray tracing

Two ways to train the RGB student policy when ray-traced rendering is too slow to
sit inside the training loop:

- **A. Offline BC** — pre-render a fixed dataset of generated frames, then behaviour-clone on it.
- **B. Online image-model DAgger** — a warm Image3F server relabels every observation
  in the rollout, so the student trains on generated pixels while the teacher supervises actions.

Both produce a student for `walk_stand_place_grasp_turn_homie` (bottle → tray).
Code lives in `GR00T-VisualSim2Real/`; launchers and assets live here in `fast_roborender/`.

## Setup

OCI uses three separate environments because the host telemetry process, Image3F renderer,
and Isaac/GR00T trainer have incompatible runtime requirements:

| Environment | Built by | Used for |
|---|---|---|
| Host W&B venv | `scripts/scaling/oci_host_wandb_setup.sbatch` | Coordinator-side telemetry and media uploads |
| Image3F CUDA 12.8 venv | `scripts/scaling/oci_h100_image3f_setup.sbatch` | H100 Image3F serving, ONNX export, TensorRT build, TeaCache calibration, and smoke gate |
| Isaac/GR00T venv | `scripts/scaling/oci_h100_trainer_setup.sbatch` | Isaac Sim 5.1, IsaacLab, integrated GR00T trainer, and evaluation |

```bash
# Shared OCI/Lustre paths used by all three jobs.
source scripts/setup/oci_env.sh

# These environments are independent and may be submitted together.
sbatch scripts/scaling/oci_host_wandb_setup.sbatch
sbatch scripts/scaling/oci_h100_image3f_setup.sbatch
sbatch scripts/scaling/oci_h100_trainer_setup.sbatch

# Read-only aggregate preflight after all three jobs finish.
bash scripts/setup/check_env.sh
```

`check_env.sh` is read-only and prints PASS/WARN/FAIL for the three OCI environments,
Image3F adapter and foundation weights, the H100 TensorRT engine, anchor pool, expert
labeler, and the `general_cs_infra` Slurm account. WARN means “fine unless you need that
stage.” Exit status is non-zero only on hard failures.

### Setup gotchas

- **Engines are per GPU architecture and TensorRT version.** OCI production builds an
  H100 `sm90` engine with TensorRT 10.13.3.9. A different GPU/software stack or LoRA
  requires a fresh ONNX export and engine build. Nothing is shipped prebuilt —
  `assets/generated/` starts empty by design.
- **The bundles hash-verify their own source tree.** Never leave edits in a bundle: patch,
  run, restore. `PYTHONDONTWRITEBYTECODE=1` is required, since stray `.pyc` files fail the
  release-tree check, and the package-stage doctor demands an empty `assets/generated/`
  (clear any orphaned staging after a failed export).
- **Extract archives inside a Slurm job.** The login-node process reaper kills long
  unzips (a 22 GB extract died with exit 137).
- The Image3F bundle ships only the LoRA delta; foundation weights are attached via
  `ROBORENDER_MODEL_BASE`, pointing at the older portable bundle's `assets/models`.
- Model load is ~110 s with a warm page cache but can be much longer cold from Lustre, which is why
  the render server is a persistent job rather than something each run starts.

## Repository layout

```
README.md                     this file
run_raven_server.sh           render-server entrypoint (bundle-relative; stays at root)

scripts/
  setup/    oci_env.sh  check_env.sh
  scaling/  scale_plan.py  image3f_probe.py  install_gr00t_integration.py
            oci_host_wandb_setup.sbatch  oci_h100_image3f_setup.sbatch
            oci_h100_trainer_setup.sbatch  oci_packed_256x300.sbatch
            submit_oci_packed.sh  submit_production.sh  submit_8h_256env.sh
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

GR00T-VisualSim2Real/         trainer, configs, simulator (own git repo, pinned
                              OCI integration branch)
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
| **Image3F finetune (image model)** | `$ROBORENDER_STAGE_ROOT/assets/roborender_image_model_accelerated`, adapter `rgb030-depth030-step-5000.safetensors` | **single 3-view frame** | ~192 ms/frame warm | **B** — the only option for in-loop relabelling |

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
`scripts/anchors/videogen_prompts.py` / `scripts/anchors/rich_prompts.py` for randomized
scenes. `scripts/data/render_shard.sbatch` splits the dataset across several servers.

**What the renderer writes.** Each chunk gains

```
rgb_gen                (81, 108, 192, 3)   the primary view, for single-view policies
rgb_gen__head_color    (81, 108, 192, 3)   per-view, for multi-view policies
rgb_gen__hand_left     (81, 108, 192, 3)
rgb_gen__hand_right    (81, 108, 192, 3)
```

The per-view keys are what a 3-view offline BC policy tiles. Resume is view-aware: a chunk
counts as done only when `rgb_gen` **and** every requested `rgb_gen__<view>` is present, so
pointing the renderer at a dataset rendered before per-view support re-renders it rather
than skipping it.

**Re-rendering an existing dataset for multi-view.** Start one teacher server per shard,
each writing its own `teacher_endpoint_<i>.txt`, then launch the shards:

```bash
LORA=.../droidnew_continue_lora/step-5294.safetensors
for i in 0 1 2 3; do
  sbatch --time=12:00:00 --job-name=teach_srv_$i \
    --export=ALL,ENDPOINT_FILE=$PWD/teacher_endpoint_$i.txt,RAVEN_SERVER_ARGS="--teacher_lora $LORA" \
    scripts/server/raven_server.sbatch
done
# once every endpoint file names the node its server runs on:
for i in 0 1 2 3; do
  sbatch --export=ALL,SHARD=$i,NSHARDS=4 scripts/data/rerender_3view.sbatch
done
```

`rerender_3view.sbatch` is finite (no `--follow`), so each shard exits when its slice is
done. ~32 s per chunk with the 50-step teacher, so 2000 chunks over 4 shards is roughly
4.5 h. Budget ~10 MB per chunk of extra storage for the three per-view arrays.

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

**The pool.** `$ROBORENDER_STAGE_ROOT/assets/anchor_pool_2048` — 2048 anchors, each an independent scene
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
  sbatch --export=START=$((SH*683)),END=$(((SH+1)*683)),OUT="$ROBORENDER_STAGE_ROOT/assets/anchor_pool_2048",TEACHER_LORA="$TEACHER_LORA" scripts/anchors/anchor_pool.sbatch
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

Image3F relabels observations inside the rollout, so the student is trained on generated
pixels over a live state distribution rather than a fixed dataset.

**Who drives the rollout is a config choice, and the default is the teacher.**
`videogen.rollout_action_source: teacher` (the default, and what every round so far used)
overrides the student's actions with the teacher's each step:

```python
# distill_trainer_videogen.py
if self._in_train_rollout and self.vg_rollout_action_source == "teacher" \
   and "gt_actions" in policy_state_dict:
    policy_state_dict["actions"] = policy_state_dict["gt_actions"].clone()
```

So the student learns on the *teacher's* state distribution — DAgger at β=1, i.e. on-policy
BC over teacher-visited states, not the student-state correction that classic DAgger
performs. The 69.3% result was obtained this way.

`rollout_action_source: student` makes the student drive, which is the classic β=0 setting.
It used to carry a caveat — the student saw the previous chunk's last generated frame as
stale vision — but that path is gated to `videogen.mode: online`. In `image_dagger` the
relabel happens per step *before* the policy acts, so a student-driven rollout sees fresh
generated frames. It is untested at scale; every result to date uses the teacher-driven default.

For the annealed mixture the paper actually prescribes, set `rollout_action_source: beta`.
Each environment then follows the teacher with probability β at **each decision** (Ross et
al. sample the mixture per decision, not per trajectory), with β annealed from `beta_start`
to `beta_end` over `beta_decay_iters`. The paper starts at β=1 and decays toward 0 —
expert-driven while the student is still random, student-driven by the end — and its
recommended practical setting is β=1 on the first iteration and 0 thereafter. A fixed
`beta_schedule: constant` with `beta_start: 0.5` gives an even split, but never collects
the student's undiluted state distribution, which is the point of DAgger. β affects
training rollouts only; evaluation always runs the student.

```bash
# The packed launcher starts both renderer and trainer steps inside one allocation.
source scripts/setup/oci_env.sh
DRY_RUN=1 bash scripts/scaling/submit_oci_packed.sh
bash scripts/scaling/submit_oci_packed.sh
```

### Observation mixing

`MIX` is the fraction of environments left on **ray-traced** observations
(`videogen.raytraced_mix_prob`):

| `MIX` | Regime | Renderers for 16 envs | Notes |
|---|---|---|---|
| `0.0` | **fully image-generated** | ~3 | every env relabelled; trains purely on the generated distribution |
| `0.5` | mixed (historical) | ~2 | what round 4 used → 69.3%; hedges against renderer artefacts |
| `1.0` | fully ray-traced | 0 | no relabelling; plain DAgger, useful as a control |

`scripts/train/dagger_train.sbatch` validates `MIX ∈ [0,1]`, discovers endpoint files whose host still runs
an `i3f_warm_server` (a stale endpoint otherwise kills the run hours in), refuses to start if
relabelling is requested with no live renderer, and warns when there are fewer servers than
`ceil(envs × (1 − MIX) / 5)`. Too few servers does not fail — it serialises the rollout, so
watch `Iteration time` (expect ~145 s at 16 envs / 3 servers / `MIX=0.5`).

That table describes the earlier small-run launcher. OCI production fixes `MIX=0`, starts
the student from base initialization (`CKPT=none`), and validates all 52 renderer endpoints
inside the packed job before training.

### Config that matters

`gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml`:

| Key | Value | Why |
|---|---|---|
| `algo.config.value_loss_coef` / `entropy_coef` | `0.0` | pure BC; RL terms with an untrained value head corrupt the policy |
| `videogen.mode` | `image_dagger` | per-step 3-view relabel in `distill_trainer_videogen.py` |
| `videogen.raytraced_mix_prob` | via `MIX` | see above |
| `videogen.image3f_endpoint_files` | one per renderer | envs are sharded round-robin across them |
| `videogen.anchor_pool_dir` / `anchor_index` | `anchor_pool_2048` / `0` | fixed anchor per episode; chaining on own outputs collapses |
| `videogen.rollout_action_source` | `beta` in production | per-decision expert/student mixture |
| `videogen.beta_schedule` | `linear` | `constant` / `linear` / `exponential`, used when the source is `beta` |
| `videogen.beta_start` / `beta_end` | `1.0` / `0.0` | DAgger mixture endpoints |
| `videogen.beta_decay_iters` / `beta_decay_rate` | `500` / `0.99` | linear horizon / exponential rate |
| `num_steps_per_env` | `300` in production | rollout horizon per outer DAgger iteration |

Historically, round 4 warm-started from round 3 and reached `dagger_bc_loss` 0.146 within
~900 iterations (round 3 ended at 0.67 and finished at 0.134). The OCI campaign does not
reuse that policy: its ImageNet-pretrained ResNet-18 encoder and policy heads start from the
base model path, while the expert checkpoint is used only to label actions.

### Production scale: 256 environments × 300 rollout steps

Production is one packed Slurm allocation: **8 nodes × 8 H100 GPUs = 64 H100s**. Inside that
allocation, one concurrent step runs 52 persistent Image3F renderer tasks (up to seven per
node), and a second step runs eight synchronized trainer ranks (one per node). Four GPUs remain
as placement slack. There are no independently scheduled renderer or trainer jobs.

Each trainer rank owns 32 Isaac environments, giving 256 globally. `num_envs` is rank-local in
GR00T, so passing 256 to every rank would accidentally create 2,048 environments. Every outer
iteration collects 256 × 300 = 76,800 transitions. With one minibatch, one learning epoch,
microbatch 1, and gradient accumulation 32, it performs one synchronized optimizer update over
all 76,800 transitions.

Production fixes `MIX=0`, so all 256 environments receive Image3F observations at every tick.
The reference policy is FT-FIXED (`REFERENCE_LAG=0`): one teacher-rendered RGB triplet is sampled
for each episode and reused on every frame. Generated frames are never fed back as later
references. The 52 endpoints are run-scoped and must all report protocol, release, adapter,
sampler, TeaCache, TensorRT, and runtime identities matching the launch contract before the first
rollout.

The core trainer is a separate, pinned GR00T checkout. Public NVlabs GR00T does not have
`distill_trainer_videogen.py`, `image3f_frame_client.py`, or `wsdpt_image_dagger.yaml`. The
installer patches the known parallel trainer to pass explicit episode/step IDs and installs
the versioned client. Run it only after checking out that exact fork, review both changes, then
commit them in the GR00T repository so production sees a clean, pinned revision:

```bash
python scripts/scaling/install_gr00t_integration.py \
  --gr00t-root GR00T-VisualSim2Real --install

# inspect the exact 256x300 capacity calculation
python scripts/scaling/scale_plan.py

# validate the single packed command without submitting
DRY_RUN=1 bash scripts/scaling/submit_oci_packed.sh

# submit the packed 8-node allocation
bash scripts/scaling/submit_oci_packed.sh

# Compatibility entry points resolve to the same packed launcher.
bash scripts/scaling/submit_production.sh
bash scripts/scaling/submit_8h_256env.sh
```

The wall time covers renderer startup, endpoint validation, training, telemetry drain, and
cleanup inside the same allocation. The packed coordinator owns both `srun` steps: if a renderer
dies while training is live, it signals the trainer and fails the job; on any exit it terminates
both steps. Slurm sends `USR1` five minutes before the allocation limit, allowing the trainer to
retain its latest periodic checkpoint and drain telemetry before the job ends. Submission and
packed manifests record the single job ID, both repository revisions, asset identities, and the
52+8+4 GPU layout.

The production contract fixes 256 environments, 300 rollout steps, eight trainer ranks,
52 renderers, `MIX=0`, fixed anchors, and fresh student initialization. Useful campaign-level
overrides are `NUM_ROLLOUT_ITERATIONS`, `SAVE_FREQUENCY`, `PROJ`, `EXP`, `RUN_ID`, and
`RUN_WALLTIME`; source and asset identities remain pinned. `ROLLOUT_STEPS=300` is the collection
horizon of each outer iteration, not an optimizer-batch size or run length.

Production behavior is explicit and recorded in both manifests. It uses per-decision β mixing,
annealed linearly from 1.0 to 0.0 over 500 rollout iterations: the expert dominates while the
fresh student is random, and the student increasingly drives the visited-state distribution.
The legacy parent-trainer mixer remains disabled. `NUM_LEARNING_EPOCHS=1`, one minibatch,
`VG_PRIMARY_VIEW=head_color`, `value_loss_coef=0.0`, and `entropy_coef=0.0` are pinned. Image3F
generates all three synchronized views, while the current actor consumes the selected head view.
Runtime assertions reject any composed Hydra configuration that disagrees with this contract.

### Explicit production gates

| Gate | Must pass before/while running |
|---|---|
| Source and assets | Clean, run-pinned `fast_roborender` and integrated GR00T revisions; exact expert-labeler SHA; base student initialization; versioned Image3F client and telemetry integration |
| Scale and behavior | 8 nodes × 8 H100s; 8 ranks × 32 environments = 256; 300-step rollout; 52 renderers + 4 slack; `MIX=0`; FT-FIXED lag 0; β 1→0 over 500 iterations; pure DAgger BC |
| Renderer provenance | All 52 live endpoints report the exact fine-tuned release/adapter SHA, five-step sampler, TeaCache 0.3, CFG 1.0, CUDA graph setting, and a load-verified TensorRT engine compatible with the serving GPU/software stack |
| W&B observability | Online personal workspace authenticated as `ldu@nvidia.com`; stable resumable sidecar run; scalar, condition/generated image, and MP4 producers ready; repeated upload/encoding errors terminate training |
| Runtime semantics | The composed trainer agrees with every behavior, primary-view, epoch, and loss-coefficient assertion before its first rollout |
| Scientific result | Baseline/early/mid/final fixed-seed evaluation is complete; the final policy improves success by at least 2 absolute points and the paired 95% bootstrap interval has a lower bound above zero |

The packed topology requires eight OCI H100 nodes at once. Source alone is not sufficient: the
integrated GR00T checkout, expert labeler, Image3F adapter and foundation weights, anchor pool,
target-built H100 TensorRT engine, three ready environments, and W&B authentication must all be
present. A failed gate never becomes a nominal production run.

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

The packed production launcher intentionally submits only the training allocation; it does not
silently create follow-on evaluation jobs. After training, the fixed-seed evaluation entrypoints
can compare a declared baseline, early, midpoint, and final checkpoint on the same eight Isaac
seed blocks with 32 environments per block. Each environment contributes exactly one episode,
giving 256 matched episodes per checkpoint without letting fast-terminating policies advance the
reset RNG to different scenarios. Because production starts a fresh student, the baseline must be
declared explicitly (for example, a captured iteration-zero policy) rather than inferred from a
warm-start checkpoint. Baseline and reference-config hashes are pinned before evaluation.
Evaluation atomically rebuilds each staged checkpoint, removes only environment, optimizer,
and scheduler state, and records the actual staged checkpoint/config hashes before reset. A strict
in-tree adapter captures the mapping returned by `trainer.eval()` and writes it atomically; the
gate does not rely on a particular trainer variant happening to emit `metrics_eval.json`.

The machine-readable report is
`<training_run>/fixed_seed_eval/fixed_seed_improvement_report.json`. The gate passes only when
final-minus-baseline success is at least two absolute percentage points and the lower bound of
its paired seed-block bootstrap 95% interval is above zero. Reward and its paired interval are
reported as a diagnostic but are not silently mixed into the success criterion. Training loss
is explicitly excluded as evidence of closed-loop improvement. Baseline/early/mid/final success,
Wilson intervals, reward intervals, and the gate result are logged to a separate evaluation run
in the same personal W&B project. This post-run scientific gate does not change the lifecycle or
ownership of the completed packed production job.

Before the full job, validate environment imports, the H100 renderer smoke test, endpoint
identity checks, and a bounded distributed rollout. Acceptance requires exact global transition
counts, finite BC loss, balanced endpoint request counts, zero render errors, correct FT-FIXED
anchor resets, and synchronized policy checkpoints across ranks.

## Multi-view policies

By default the policy sees one camera (the ego/head view). Image3F renders three
synchronized views regardless, so the wrist views were being generated and discarded.

Set `cameras.policy_views` and the matching `policy_view_count` and the views are tiled
vertically into a single image:

```
[ head_color ]  108x192
[ hand_left  ]  108x192   ->  vision_obs (324, 192, 3)  ->  ResNet18
[ hand_right ]  108x192
```

Ready-made configs: `wsdpt_image_dagger_3view` and `wsdpt_videogen_offline_bc_3view`.

```bash
sbatch --export=ALL,MIX=0.0,CKPT=none,NUM_ENVS=4 scripts/train/dagger_train_3view.sbatch
```

Ray-traced envs take the tiles from the conditioning cameras (already rendered for the
renderer, just downsampled), generated envs from Image3F — identical layout either way.

Things that follow from the taller observation:

- **Single-view checkpoints cannot be warm-started.** The vision weights have the wrong
  shape, so launch with `CKPT=none`. The 57.3% and 69.3% baselines do not transfer, and a
  3-view baseline has to be re-established.
- **Offline BC needs the dataset re-rendered.** Chunks must carry `rgb_gen__<view>` per
  view; older ones have only the primary `rgb_gen`. `offline_render_dataset.py` writes the
  per-view keys and its resume check treats a chunk as done only when every requested view
  is present, so pointing it at an old dataset re-renders it.
- `policy_view_count` must equal `len(policy_views)`. The obs dims scale by the former and
  the trainer tiles by the latter; a mismatch mis-reshapes rather than erroring cleanly.
- Renderer load scales with relabelled envs, not views — one server per ~5 envs still holds,
  but each request now returns three views instead of one.

Verified end to end (`MIX=0.0`, 4 envs, from scratch): history buffer `(324, 192, 3)`,
storage `obs_dim=186624`, two learning iterations with `dagger_bc_loss` 1.089 -> 0.873.
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
`i3f_warm_server` running on it, which is what a stale endpoint looks like. It logs to the
OCI/Lustre run log directory, prints a
one-line `RESULT ...` summary, and cleans up its staging directory.

Reference configs it stages against:

- `raytraced` → `logs_rl/videogen_offline_bc/videogen_offline_bc-20260721_204447/config.yaml`
  — the environment every number in [Results](#results) was measured in.
- `generated` → `logs_rl/image_dagger_r4/image_dagger_run4-20260814_131202/config.yaml`
  — same environment plus the `videogen`/`image_dagger` block that relabelling needs.

**One eval saturates one renderer** (~52 episodes/h). Two evals sharing a server each run at
half speed and overrun the job limit with no output, since the success rate is only printed
at the end. Give each concurrent `OBS=generated` eval its own server and `ENDPOINT`.

## Running on another machine

The trainer, configs and simulator changes live in `GR00T-VisualSim2Real/`, which is a
checkout of the public NVlabs repo with our work on the `videogen-image-dagger` branch.
A fresh clone elsewhere has none of it, and the preflight will refuse to run.

`transfer/` holds the branch as patches against public revision `92bf0863`, which is an
ancestor of the branch, so they apply without a rebase:

```bash
cd GR00T-VisualSim2Real
git am ../transfer/0001-*.patch ../transfer/0002-*.patch ../transfer/0003-*.patch
```

0001 = videogen pipeline (trainer, render clients, conditioning cameras, scene walls,
configs) · 0002 = three-view policy · 0003 = DAgger beta schedule.

The patches carry code only. Also needed on the target machine, none of it in this repo:

| Asset | Where it comes from |
|---|---|
| Image3F bundle + adapter | `$ROBORENDER_STAGE_ROOT/assets/roborender_image_model_accelerated` |
| Foundation weights | `ROBORENDER_MODEL_BASE`, see [Setup](#setup) |
| TensorRT engines | built locally per GPU arch — never shipped |
| Anchor pool | `scripts/anchors/anchor_pool.sbatch` |
| Rendered dataset | `scripts/data/` (offline BC only) |
| Launchers and this README | `fast_roborender/` |

Every launcher in `scripts/` now targets the OCI cluster (`general_cs_infra`,
`grizzly` for the packed 8-node job, `backfill_block1` for single-node GPU work,
`cpu_short` for host-only utilities, and the shared Lustre stage root). The current
`grizzly` QoS caps each packed stage at four hours. On a workstation with no Slurm, run the training
command directly — the launchers are thin wrappers around it:

```bash
cd GR00T-VisualSim2Real
HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    gr00t/rl/train_agent_trl.py \
    +exp=loco_manip/wsdpt_image_dagger_3view \
    num_envs=4 headless=True \
    algo.config.videogen.raytraced_mix_prob=0.0 \
    algo.config.videogen.anchor_pool_dir=/path/to/anchor_pool \
    algo.config.videogen.image3f_endpoint_files=[/path/to/endpoint.txt] \
    experiment_name=dagger_3view project_name=my_project
```

Omit `checkpoint=` entirely for a 3-view run: single-view weights cannot be loaded.
Start the renderer first with `runtime/image3f_deploy.py` behind
`scripts/server/image3f_server.py`, and point `image3f_endpoint_files` at the file it
writes.

`bash scripts/setup/check_env.sh` reports which assets are missing. Its paths default to
the OCI/Lustre layout and are overridable:

```bash
ROOT=/your/checkout IMAGE3F=/your/image3f_bundle VIRAL_PY=$CONDA_PREFIX/bin/python \
  bash scripts/setup/check_env.sh
```

The environment installer is portable as well: set `CONDA`, `ROOT`, and optionally
`VIRAL_ENV_NAME`, `VIRAL_PY`, or `ROBORENDER_CACHE_ROOT` before running
`bash scripts/setup/setup_viral_env.sh`.

`python3 scripts/setup/check_paths.py` verifies no script references a file that moved.

> Note for reviewers: `archive/` contains superseded one-off scripts, including the
> `patch_*.py` snapshots that were applied once to produce the current code. **They are not
> the integration** — read `distill_trainer_videogen.py` and the simulator's
> `get_policy_view_stack` instead. See `archive/README.md`.

## Depth control recipe

Depth is what the renderer conditions on, and the exact h5_v4 parameters matter. They live
in the experiment config (`videogen.depth_h5v4_per_view`) and are applied by
`_h5v4_control` in the trainer:

```yaml
head_color: {raw_clip: [0.05, 10.0], inv_clip: [1.0, 40.0], norm_range: [0.05, 2.6]}
hand_left:  {raw_clip: [0.0, 1.5],  inv_clip: [0.7, 100.0], raw_pct: [1, 99]}
hand_right: {raw_clip: [0.0, 1.5],  inv_clip: [0.7, 100.0], raw_pct: [1, 99]}
```

`norm_range` on the head view is load-bearing: with fixed normalization bounds a close
object no longer anchors the range to white, so object shape survives. Views without
`norm_range` fall back to per-video min/max, and views without `raw_pct` skip percentile
clipping (the built-in default carries neither).

Denoising steps, measured with the correct recipe (6 frames, 2 chunks, FT-FIXED, fixed
seed; PSNR against the 50-step video teacher):

| steps | ms | PSNR | SSIM |
|---|---|---|---|
| **5** (production) | 212 | **17.91** | 0.792 |
| 20 | 689 | 15.01 | 0.659 |
| 50 | 1598 | 14.53 | 0.634 |

More denoising is worse here: geometry stays correct and textures sharpen, but the output
drifts off the anchor's colours and materials, which is exactly what FT-FIXED depends on.
The same ranking held under the earlier incorrect depth, 0.7-1.2 dB lower across the board.

**Offline analysis must use `tools/depth_recipe.py`**, which loads these params from the
config and reproduces `_h5v4_control` byte-for-byte (`tools/test_depth_recipe.py` asserts
equality against the trainer's own code). Hand-copying the recipe into a script is how the
A/B evals ended up inverting raw metric depth and min-maxing per frame: a 37/255 mean
difference on real head-view data, 61% of pixels off by more than 32 levels, understating
every arm by 0.7-1.2 dB. Training was never affected -- it always went through the config.

## Gotchas

- **Scene walls must not be collidable.** `cameras.add_scene_walls` gives the head camera a
  real background instead of void, but collidable walls sit in the robot's walking path and
  cost ~40 points (16.7% collidable / 56.0% none / 60.0% visible-but-non-collidable). The
  `cameras.wall_collisions` flag (default `False`, `isaacsim.py`) keeps the backdrop and
  removes the obstacle. Leave it off.
- **Stale endpoints.** `image3f_queue*/READY` and `image3f_server_endpoint*.txt` persist
  across jobs. A readiness check that only tests file existence will fire on a dead server
  from a previous run. Verify the endpoint's hostname matches the server job's node.
- **Shared storage stalls.** Check `ps -o wchan` and the Lustre client state when a job
  runs with an empty log; never work around this by writing large artifacts to the root disk.
- **Slurm.** `--nodelist=A,B` requests *all* listed nodes. OCI production uses
  `--gpus-per-node=8` for the packed run and `--gpus-per-node=1` for single-GPU helpers.
  Hydra needs a `+` prefix for keys absent from
  the base config. Training runs end in `TIMEOUT` at their limit; that is normal, and
  `--dependency=afterany` (not `afterok`) is what chains an eval behind them.
