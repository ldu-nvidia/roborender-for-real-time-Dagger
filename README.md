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

## A. Offline BC

```bash
# 1. collect depth/mask chunks from teacher rollouts
sbatch collect_data.sbatch                  # +exp=loco_manip/wsdpt_videogen_collect

# 2. render RGB for the chunks (50-step teacher video model, or Image3F)
sbatch offline_render.sbatch                # -> offline_render_dataset.py

# 3. behaviour-clone on the rendered dataset
sbatch offline_bc_train.sbatch              # +exp=loco_manip/wsdpt_videogen_offline_bc
```

`offline_bc_train.sbatch` takes `NUM_ENVS` (default 16). Checkpoints land in
`logs_rl/videogen_offline_bc/<run>/`.

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

```bash
# 1. teacher anchor pool (one content-rich RGB triplet per episode; FT-FIXED needs it)
sbatch anchor_pool.sbatch                   # render_anchor_pool.py + rich_prompts.py

# 2. warm renderers — ONE PER ~5 ENVS, each with its own endpoint file and queue
for i in 1 2 3; do
  EP=.../image3f_server_endpoint${i:+_$i}.txt; Q=.../image3f_queue${i:+_$i}
  sbatch --time=3-00:00:00 --export=ALL,ENDPOINT_FILE=$EP,IMAGE3F_QUEUE=$Q image3f_server.sbatch
done

# 3. train (wait until every endpoint file names the node its job is running on)
sbatch image_dagger_r4.sbatch               # +exp=loco_manip/wsdpt_image_dagger
```

Key config (`gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml`):

| Key | Value | Why |
|---|---|---|
| `algo.config.value_loss_coef` / `entropy_coef` | `0.0` | pure BC; RL terms with an untrained value head corrupt the policy |
| `videogen.mode` | `image_dagger` | per-step 3-view relabel in `distill_trainer_videogen.py` |
| `videogen.image3f_endpoint_files` | one path per renderer | envs are sharded round-robin across them |
| `videogen.anchor_pool_dir` / `anchor_index` | `anchor_pool_2048` / `0` | fixed anchor per episode (FT-FIXED); chaining on own outputs collapses |
| `videogen.raytraced_mix_prob` | `0.5` | fraction of envs left on ray-traced RGB |
| `num_steps_per_env` | `81` | one BC update per video-model clip length |

Throughput: ~145 s/iteration at 16 envs / 3 renderers. Warm-start from the previous
round's `last.pt` — round 4 started from round 3 and reached `dagger_bc_loss` 0.146
within ~900 iterations (round 3 ended at 0.67).

## Evaluation — read this first

**`eval_agent_trl.py` loads `config.yaml` from the checkpoint's own directory.** A DAgger
run's config carries simulator settings that differ from the baseline's, so scoring a
DAgger checkpoint against its own config compares two different environments and the
number is meaningless. This invalidated three rounds of results before it was caught.

Always stage the checkpoint next to the **baseline's** config:

```bash
D=diag_eval; mkdir -p $D
cp <run>/last.pt $D/last.pt
cp logs_rl/videogen_offline_bc/<baseline_run>/config.yaml $D/config.yaml
python gr00t/rl/eval_agent_trl.py +checkpoint=$D/last.pt \
    +num_envs=4 +headless=True \
    algo.config.eval.num_eval_episodes=300 \
    +simulator.config.render_results=true
```

- Default is 150 episodes (SE ≈ 4 pts). Use **300** for any number you intend to quote (SE ≈ 2.7).
- To evaluate on generated observations instead, set `IMAGE_DAGGER_EVAL_RELABEL=1` with a
  live renderer and `raytraced_mix_prob: 0.0`.
- **One eval saturates one renderer** (~52 episodes/h). Two evals sharing a server each
  run at half speed and overrun the 3 h job limit with no output — the success rate is
  only printed at the end.

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
