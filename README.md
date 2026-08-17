# Vision-student training without ray tracing

Two ways to train the RGB student policy when ray-traced rendering is too slow to
sit inside the training loop:

- **A. Offline BC** — pre-render a fixed dataset of generated frames, then behaviour-clone on it.
- **B. Online image-model DAgger** — a warm Image3F server relabels every observation
  in the rollout, so the student trains on generated pixels while the teacher supervises actions.

Both produce a student for `walk_stand_place_grasp_turn_homie` (bottle → tray).
Code lives in `GR00T-VisualSim2Real/`; launchers and assets live here in `fast_roborender/`.

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
That is what `anchor_pool.sbatch` produces (2048 anchors, `rich_prompts.py`). Conditioning
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
sbatch collect_data.sbatch                  # +exp=loco_manip/wsdpt_videogen_collect
                                            # NUM_ENVS (default 32)
```

Chunks land in `videogen_dataset/` as `chunk_<jobid>_<NNNNNNN>.npz`, each holding
`depth_raw__<view>`, `depth_ctrl__<view>`, `rgb_robot__<view>` for `head_color`,
`hand_left`, `hand_right`, plus the prompt. The production set is ~2000 chunks / 31 GB.

**2. Render RGB.** A standing render server converts depth+mask into photoreal RGB.
Start the server first, then stream the dataset through it:

```bash
sbatch raven_server.sbatch                  # or teacher_anchor_render.sbatch for the 50-step teacher
sbatch offline_render.sbatch                # -> offline_render_dataset.py --follow
                                            # RENDER_VIEWS, RENDER_ARGS
```

`--follow` keeps the renderer consuming chunks as collection produces them, so steps 1 and 2
can overlap. Scene text comes from `render_prompt.txt` (single fixed scene) or
`videogen_prompts.py` / `rich_prompts.py` for randomized scenes. `render_shard.sbatch` splits
the dataset across several servers.

Depth is normalized with the `h5_v4` recipe before it reaches the model
(`renormalize_depth.py`); per-view clip ranges differ between the head and wrist cameras.

## A. Offline BC

Behaviour-clone on the rendered dataset — no renderer in the loop, so this is the cheap path.

```bash
sbatch offline_bc_train.sbatch              # +exp=loco_manip/wsdpt_videogen_offline_bc
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

**Building it.** `render_anchor_pool.py` is shardable; `anchor_pool.sbatch` starts a teacher
server (LoRA path is required — the server dies instantly without it) and renders a range:

```bash
# START/END/OUT are read from the environment; the render server's port is derived
# from the job id inside the script, so parallel shards never collide.
for SH in 0 1 2; do
  sbatch --export=ALL,START=$((SH*683)),END=$(((SH+1)*683)),OUT=/vision/u/ravenh/anchor_pool_2048          anchor_pool.sbatch
done
```

Prompts come from `rich_prompts.py::build_rich_prompt(seed)`, which composes walls (16) ×
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
sbatch anchor_pool.sbatch

# 2. warm renderers: ~1 per 5 relabelled envs, each with its own endpoint file and queue
for i in '' _2 _3; do
  sbatch --time=3-00:00:00 \
    --export=ALL,ENDPOINT_FILE=$PWD/image3f_server_endpoint$i.txt,IMAGE3F_QUEUE=$PWD/image3f_queue$i \
    image3f_server.sbatch
done

# 3. train
sbatch --export=ALL,MIX=0.5 dagger_train.sbatch
```

### Observation mixing

`MIX` is the fraction of environments left on **ray-traced** observations
(`videogen.raytraced_mix_prob`):

| `MIX` | Regime | Renderers for 16 envs | Notes |
|---|---|---|---|
| `0.0` | **fully image-generated** | ~3 | every env relabelled; trains purely on the generated distribution |
| `0.5` | **mixed** (default) | ~2 | what round 4 used → 69.3%; hedges against renderer artefacts |
| `1.0` | fully ray-traced | 0 | no relabelling; plain DAgger, useful as a control |

`dagger_train.sbatch` validates `MIX ∈ [0,1]`, discovers endpoint files whose host still runs
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

## Evaluation — read this first

**`eval_agent_trl.py` loads `config.yaml` from the checkpoint's own directory.** A DAgger
run's config describes a different environment than the baseline's, so scoring a DAgger
checkpoint out of its run directory compares two different worlds and the number means
nothing. This invalidated three rounds of results before it was caught.

`eval_policy.sbatch` exists so that cannot happen: it always stages the checkpoint beside an
explicit reference config.

```bash
# headline number: ray-traced obs, baseline config, 300 episodes
sbatch --export=ALL,CKPT=<run>/last.pt eval_policy.sbatch

# quick check
sbatch --export=ALL,CKPT=<run>/last.pt,EPISODES=50 eval_policy.sbatch

# generated obs (needs a live renderer)
sbatch --export=ALL,CKPT=<run>/last.pt,OBS=generated eval_policy.sbatch
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
