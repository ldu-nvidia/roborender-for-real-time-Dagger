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
setup_viral_env.sh            conda env for the trainer

scripts/
  setup/    setup_viral_env.sh  check_env.sh
  data/     collect_data.sbatch  offline_render.sbatch  render_shard.sbatch
            offline_render_dataset.py  renormalize_depth.py
  anchors/  anchor_pool.sbatch  render_anchor_pool.py  rich_prompts.py
            teacher_anchor_render.sbatch  render_teacher_fullres.py  videogen_prompts.py
  train/    offline_bc_train.sbatch  dagger_train.sbatch  image_dagger_*.sbatch
  eval/     eval_policy.sbatch  eval_latest.sbatch  eval_student.sbatch
  server/   image3f_server.py  image3f_server.sbatch  raven_server.sbatch
            image3f_ft_setup.sbatch  image3f_ft_eval.sbatch

tools/      benchmarks, sample/preview builders, the 5-vs-50-step A/B (step_ab*.py)

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
| `videogen.rollout_action_source` | `teacher` | who drives the rollout: `teacher` (β=1), `student` (β=0), `beta` (annealed) |
| `videogen.beta_schedule` | `linear` | `constant` / `linear` / `exponential`, used when the source is `beta` |
| `videogen.beta_start` / `beta_end` | `1.0` / `0.0` | DAgger mixture endpoints |
| `videogen.beta_decay_iters` / `beta_decay_rate` | `500` / `0.99` | linear horizon / exponential rate |
| `num_steps_per_env` | `81` | one BC update per video-model clip length |

Warm-start from the previous round's `last.pt`: round 4 started from round 3 and reached
`dagger_bc_loss` 0.146 within ~900 iterations (round 3 ended at 0.67, and finished at 0.134).

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
| Image3F bundle + adapter | `/vision/u/ravenh/roborender_image_model_accelerated` |
| Foundation weights | `ROBORENDER_MODEL_BASE`, see [Setup](#setup) |
| TensorRT engines | built locally per GPU arch — never shipped |
| Anchor pool | `scripts/anchors/anchor_pool.sbatch` |
| Rendered dataset | `scripts/data/` (offline BC only) |
| Launchers and this README | `fast_roborender/` |

Every launcher in `scripts/` is an sbatch script written for this cluster (viscam
partitions, `/svl` and `/vision` paths). On a workstation with no Slurm, run the training
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
the sc layout and are overridable:

```bash
ROOT=/your/checkout IMAGE3F=/your/image3f_bundle VIRAL_PY=$CONDA_PREFIX/bin/python \
  bash scripts/setup/check_env.sh
```

`python3 scripts/setup/check_paths.py` verifies no script references a file that moved.

> Note for reviewers: superseded one-off diagnostics and the historical
> `patch_*.py` snapshots were intentionally omitted from this source-only snapshot.
> **They are not the integration** — read `distill_trainer_videogen.py` and the
> simulator's `get_policy_view_stack` instead. Their history remains in the upstream
> repository lineage.

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
- **NFS hangs.** viscam10 and viscam12 have wedged `cp` in `handle_async_copy` for an hour
  with the destination already fully written. `--exclude=viscam10,viscam12` for eval jobs;
  check `ps -o wchan` on the node when a job runs with an empty log.
- **Slurm.** `--nodelist=A,B` requests *all* listed nodes — use `--gres=gpu:<type>:1 --nodes=1`.
  `viscam-hi` preempts `viscam`/`sc-loprio`. Hydra needs a `+` prefix for keys absent from
  the base config. Training runs end in `TIMEOUT` at their limit; that is normal, and
  `--dependency=afterany` (not `afterok`) is what chains an eval behind them.
